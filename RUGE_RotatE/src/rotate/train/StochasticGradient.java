package rotate.train;

import model.struct.Matrix;
import model.struct.Triple;

public class StochasticGradient {
	private final Triple triple;
	private final double size;
	private final Matrix entityReal;
	private final Matrix entityImag;
	private final Matrix relationPhase;
	private final Matrix entityRealGradient;
	private final Matrix entityImagGradient;
	private final Matrix relationPhaseGradient;
	private final double lambda;
	private final double label;
	private final double margin;

	public StochasticGradient(Triple triple, double size,
			Matrix entityReal, Matrix entityImag, Matrix relationPhase,
			Matrix entityRealGradient, Matrix entityImagGradient,
			Matrix relationPhaseGradient, double lambda, double label, double margin) {
		this.triple = triple;
		this.size = size;
		this.entityReal = entityReal;
		this.entityImag = entityImag;
		this.relationPhase = relationPhase;
		this.entityRealGradient = entityRealGradient;
		this.entityImagGradient = entityImagGradient;
		this.relationPhaseGradient = relationPhaseGradient;
		this.lambda = lambda;
		this.label = label;
		this.margin = margin;
	}

	public void calculateGradient() throws Exception {
		int head = triple.head();
		int tail = triple.tail();
		int relation = triple.relation();
		int dimensions = entityReal.columns();
		double scoreGradient = (RotatE.sigmoid(
				RotatE.score(triple, entityReal, entityImag, relationPhase, margin)) - label) / size;
		double regularizationScale = 2.0 * lambda / (dimensions * size);

		for (int p = 0; p < dimensions; p++) {
			double phase = relationPhase.get(relation, p);
			double cos = Math.cos(phase);
			double sin = Math.sin(phase);
			double headReal = entityReal.get(head, p);
			double headImag = entityImag.get(head, p);
			double realResidual = headReal * cos - headImag * sin - entityReal.get(tail, p);
			double imagResidual = headReal * sin + headImag * cos - entityImag.get(tail, p);
			double realSign = Math.signum(realResidual);
			double imagSign = Math.signum(imagResidual);

			double headRealGradient = -(realSign * cos + imagSign * sin);
			double headImagGradient = realSign * sin - imagSign * cos;
			double tailRealGradient = realSign;
			double tailImagGradient = imagSign;
			double phaseGradient = realSign * (headReal * sin + headImag * cos)
					- imagSign * (headReal * cos - headImag * sin);

			entityRealGradient.add(head, p, scoreGradient * headRealGradient
					+ regularizationScale * headReal);
			entityImagGradient.add(head, p, scoreGradient * headImagGradient
					+ regularizationScale * headImag);
			entityRealGradient.add(tail, p, scoreGradient * tailRealGradient
					+ regularizationScale * entityReal.get(tail, p));
			entityImagGradient.add(tail, p, scoreGradient * tailImagGradient
					+ regularizationScale * entityImag.get(tail, p));
			relationPhaseGradient.add(relation, p, scoreGradient * phaseGradient);
		}
	}
}
