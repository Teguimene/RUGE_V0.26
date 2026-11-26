package rotate.train;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import model.struct.Matrix;
import model.struct.Rule;
import model.struct.Triple;

public class StochasticUpdate {
	private final ArrayList<Triple> positiveTriples;
	private final ArrayList<Triple> headNegativeTriples;
	private final ArrayList<Triple> tailNegativeTriples;
	private final ArrayList<Rule> rules;
	private final Matrix entityReal;
	private final Matrix entityImag;
	private final Matrix relationPhase;
	private final Matrix entityRealGradient;
	private final Matrix entityImagGradient;
	private final Matrix relationPhaseGradient;
	private final double gammaE;
	private final double gammaR;
	private final double lambda;
	private final double c;
	private final double margin;
	private final HashSet<Integer> activeEntityRows = new HashSet<Integer>();
	private final HashSet<Integer> activeRelationRows = new HashSet<Integer>();
	private final HashSet<Integer> previousEntityRows = new HashSet<Integer>();
	private final HashSet<Integer> previousRelationRows = new HashSet<Integer>();

	public StochasticUpdate(ArrayList<Triple> positiveTriples,
			ArrayList<Triple> headNegativeTriples, ArrayList<Triple> tailNegativeTriples,
			ArrayList<Rule> rules, Matrix entityReal, Matrix entityImag, Matrix relationPhase,
			Matrix entityRealGradient, Matrix entityImagGradient,
			Matrix relationPhaseGradient, double gammaE, double gammaR,
			double lambda, double c, double margin) {
		this.positiveTriples = positiveTriples;
		this.headNegativeTriples = headNegativeTriples;
		this.tailNegativeTriples = tailNegativeTriples;
		this.rules = rules;
		this.entityReal = entityReal;
		this.entityImag = entityImag;
		this.relationPhase = relationPhase;
		this.entityRealGradient = entityRealGradient;
		this.entityImagGradient = entityImagGradient;
		this.relationPhaseGradient = relationPhaseGradient;
		this.gammaE = gammaE;
		this.gammaR = gammaR;
		this.lambda = lambda;
		this.c = c;
		this.margin = margin;
	}

	private void accumulate(Triple triple, double size, double label) throws Exception {
		activeEntityRows.add(triple.head());
		activeEntityRows.add(triple.tail());
		activeRelationRows.add(triple.relation());
		new StochasticGradient(triple, size, entityReal, entityImag, relationPhase,
				entityRealGradient, entityImagGradient, relationPhaseGradient,
				lambda, label, margin).calculateGradient();
	}

	private void update(Matrix parameters, Matrix gradients, double learningRate,
			Set<Integer> activeRows, boolean wrapPhase) throws Exception {
		for (int i : activeRows) {
			for (int j = 0; j < parameters.columns(); j++) {
				double gradient = gradients.get(i, j);
				gradients.accumulatedByGrad(i, j);
				double rate = Math.sqrt(gradients.getSum(i, j)) + 1e-8;
				double value = parameters.get(i, j) - learningRate * gradient / rate;
				parameters.set(i, j, wrapPhase ? RotatE.wrapPhase(value) : value);
			}
		}
	}

	private void resetRows(Matrix gradients, Set<Integer> rows) throws Exception {
		for (int row : rows) {
			for (int column = 0; column < gradients.columns(); column++) {
				gradients.set(row, column, 0.0);
			}
		}
	}

	private void rescaleRows(Matrix gradients, Set<Integer> rows) throws Exception {
		for (int row : rows) {
			double norm = 0.0;
			for (int column = 0; column < gradients.columns(); column++) {
				double value = gradients.get(row, column);
				norm += value * value;
			}
			norm = Math.sqrt(norm);
			if (norm > 1.0) {
				for (int column = 0; column < gradients.columns(); column++) {
					gradients.set(row, column, gradients.get(row, column) / norm);
				}
			}
		}
	}

	public void stochasticIteration() throws Exception {
		resetRows(entityRealGradient, previousEntityRows);
		resetRows(entityImagGradient, previousEntityRows);
		resetRows(relationPhaseGradient, previousRelationRows);
		activeEntityRows.clear();
		activeRelationRows.clear();

		int positiveCount = positiveTriples.size();
		int headNegativeCount = headNegativeTriples.size();
		int tailNegativeCount = tailNegativeTriples.size();
		double sampleCount = Math.max(1, positiveCount + headNegativeCount + tailNegativeCount);

		for (Triple triple : positiveTriples) {
			accumulate(triple, sampleCount, 1.0);
		}
		for (Triple triple : headNegativeTriples) {
			accumulate(triple, sampleCount, 0.0);
		}
		for (Triple triple : tailNegativeTriples) {
			accumulate(triple, sampleCount, 0.0);
		}

		HashMap<String, Double> triplePosteriors = new HashMap<String, Double>();
		for (Rule rule : rules) {
			double posterior = new PosteriorRegular(
					rule, entityReal, entityImag, relationPhase, c, margin).calculatePosterior();
			Triple unseen = unseenTriple(rule);
			String key = key(unseen);
			Double previous = triplePosteriors.get(key);
			triplePosteriors.put(key, posterior + (previous == null ? 0.0 : previous));
		}

		if (!triplePosteriors.isEmpty()) {
			Iterator<Map.Entry<String, Double>> iterator = triplePosteriors.entrySet().iterator();
			while (iterator.hasNext()) {
				Map.Entry<String, Double> entry = iterator.next();
				Triple triple = parse(entry.getKey());
				double label = RotatE.sigmoid(
						RotatE.score(triple, entityReal, entityImag, relationPhase, margin))
						+ entry.getValue();
				label = Math.max(0.0, Math.min(1.0, label));
				accumulate(triple, triplePosteriors.size(), label);
			}
		}

		rescaleRows(entityRealGradient, activeEntityRows);
		rescaleRows(entityImagGradient, activeEntityRows);
		rescaleRows(relationPhaseGradient, activeRelationRows);
		update(entityReal, entityRealGradient, gammaE, activeEntityRows, false);
		update(entityImag, entityImagGradient, gammaE, activeEntityRows, false);
		update(relationPhase, relationPhaseGradient, gammaR, activeRelationRows, true);

		previousEntityRows.clear();
		previousEntityRows.addAll(activeEntityRows);
		previousRelationRows.clear();
		previousRelationRows.addAll(activeRelationRows);
	}

	private Triple unseenTriple(Rule rule) {
		if (rule.unseen() == 1) {
			return rule.fstTriple();
		}
		if (rule.unseen() == 2) {
			return rule.sndTriple();
		}
		if (rule.unseen() == 3) {
			return rule.trdTriple();
		}
		return rule.fouTriple();
	}

	private String key(Triple triple) {
		return triple.head() + "_" + triple.relation() + "_" + triple.tail();
	}

	private Triple parse(String key) {
		String[] ids = key.split("_");
		return new Triple(Integer.parseInt(ids[0]), Integer.parseInt(ids[2]),
				Integer.parseInt(ids[1]));
	}
}
