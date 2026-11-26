package rotate.train;

import model.struct.Matrix;
import model.struct.Triple;

public final class RotatE {
	private RotatE() {
	}

	public static double score(Triple triple, Matrix entityReal, Matrix entityImag,
			Matrix relationPhase, double margin) throws Exception {
		double distance = 0.0;
		int head = triple.head();
		int tail = triple.tail();
		int relation = triple.relation();

		for (int p = 0; p < entityReal.columns(); p++) {
			double phase = relationPhase.get(relation, p);
			double cos = Math.cos(phase);
			double sin = Math.sin(phase);
			double real = entityReal.get(head, p) * cos - entityImag.get(head, p) * sin
					- entityReal.get(tail, p);
			double imag = entityReal.get(head, p) * sin + entityImag.get(head, p) * cos
					- entityImag.get(tail, p);
			distance += Math.abs(real) + Math.abs(imag);
		}
		return margin - distance;
	}

	public static double sigmoid(double value) {
		if (value >= 0.0) {
			double exp = Math.exp(-Math.min(value, 700.0));
			return 1.0 / (1.0 + exp);
		}
		double exp = Math.exp(Math.max(value, -700.0));
		return exp / (1.0 + exp);
	}

	public static double wrapPhase(double phase) {
		double wrapped = phase % (2.0 * Math.PI);
		if (wrapped > Math.PI) {
			wrapped -= 2.0 * Math.PI;
		} else if (wrapped < -Math.PI) {
			wrapped += 2.0 * Math.PI;
		}
		return wrapped;
	}
}
