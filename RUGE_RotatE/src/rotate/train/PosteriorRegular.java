package rotate.train;

import model.struct.Matrix;
import model.struct.Rule;
import model.struct.Triple;

public class PosteriorRegular {
	private final Rule rule;
	private final Matrix entityReal;
	private final Matrix entityImag;
	private final Matrix relationPhase;
	private final double c;
	private final double margin;

	public PosteriorRegular(Rule rule, Matrix entityReal, Matrix entityImag,
			Matrix relationPhase, double c, double margin) {
		this.rule = rule;
		this.entityReal = entityReal;
		this.entityImag = entityImag;
		this.relationPhase = relationPhase;
		this.c = c;
		this.margin = margin;
	}

	private double probability(Triple triple) throws Exception {
		return RotatE.sigmoid(RotatE.score(
				triple, entityReal, entityImag, relationPhase, margin));
	}

	public double calculatePosterior() throws Exception {
		double first = probability(rule.fstTriple());
		double second = probability(rule.sndTriple());
		double confidence = c * rule.confi();

		if (rule.unseen() == 1) {
			return confidence * (second - 1.0);
		}
		if (rule.unseen() == 2) {
			return confidence * first;
		}
		if (rule.unseen() == 3) {
			return confidence * first * second;
		}
		return confidence * first * second * probability(rule.trdTriple());
	}
}
