package rotate.train;

import java.util.HashMap;

import model.struct.Matrix;
import model.struct.Triple;
import model.struct.TripleSet;

public class RotatEMetricMonitor {
	private final TripleSet validationTriples;
	private final HashMap<String, Boolean> knownTriples;
	private final Matrix entityReal;
	private final Matrix entityImag;
	private final Matrix relationPhase;
	private final double margin;
	private final int maximumValidationTriples;
	public double dMRR;
	public double dHits;

	public RotatEMetricMonitor(TripleSet validationTriples,
			HashMap<String, Boolean> knownTriples, Matrix entityReal,
			Matrix entityImag, Matrix relationPhase, double margin,
			int maximumValidationTriples) {
		this.validationTriples = validationTriples;
		this.knownTriples = knownTriples;
		this.entityReal = entityReal;
		this.entityImag = entityImag;
		this.relationPhase = relationPhase;
		this.margin = margin;
		this.maximumValidationTriples = maximumValidationTriples;
	}

	public void calculateMetrics() throws Exception {
		double reciprocalRankTotal = 0.0;
		int hitsAt10Total = 0;
		int rankCount = 0;
		int numberOfEntities = entityReal.rows();

		int triplesToEvaluate = Math.min(
				validationTriples.triples(), maximumValidationTriples);
		for (int i = 0; i < triplesToEvaluate; i++) {
			if ((i + 1) % 100 == 0 || i + 1 == triplesToEvaluate) {
				System.out.println("  Validation triples ranked: " + (i + 1)
						+ "/" + triplesToEvaluate);
			}
			Triple target = validationTriples.get(i);
			double targetScore = RotatE.score(
					target, entityReal, entityImag, relationPhase, margin);

			int headRank = 1;
			for (int candidate = 0; candidate < numberOfEntities; candidate++) {
				String candidateTriple = candidate + "\t" + target.relation()
						+ "\t" + target.tail();
				if (!knownTriples.containsKey(candidateTriple)
						&& RotatE.score(new Triple(candidate, target.tail(), target.relation()),
								entityReal, entityImag, relationPhase, margin) > targetScore) {
					headRank++;
				}
			}
			reciprocalRankTotal += 1.0 / headRank;
			if (headRank <= 10) {
				hitsAt10Total++;
			}
			rankCount++;

			int tailRank = 1;
			for (int candidate = 0; candidate < numberOfEntities; candidate++) {
				String candidateTriple = target.head() + "\t" + target.relation()
						+ "\t" + candidate;
				if (!knownTriples.containsKey(candidateTriple)
						&& RotatE.score(new Triple(target.head(), candidate, target.relation()),
								entityReal, entityImag, relationPhase, margin) > targetScore) {
					tailRank++;
				}
			}
			reciprocalRankTotal += 1.0 / tailRank;
			if (tailRank <= 10) {
				hitsAt10Total++;
			}
			rankCount++;
		}

		if (rankCount == 0) {
			throw new IllegalStateException("Aucun triplet de validation à évaluer.");
		}
		dMRR = reciprocalRankTotal / rankCount;
		dHits = (double) hitsAt10Total / rankCount;
	}
}
