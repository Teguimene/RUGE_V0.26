package rotate.test;

import model.struct.Matrix;
import model.struct.Triple;
import rotate.train.RotatE;
import rotate.train.StochasticGradient;

public class RotatEModelTest {
	public static void main(String[] args) throws Exception {
		testQuarterTurnScore();
		testGradientAgainstFiniteDifferences();
		System.out.println("RotatEModelTest: OK");
	}

	private static void testQuarterTurnScore() throws Exception {
		Matrix entityReal = new Matrix(2, 1);
		Matrix entityImag = new Matrix(2, 1);
		Matrix relationPhase = new Matrix(1, 1);
		entityReal.set(0, 0, 1.0);
		entityImag.set(0, 0, 0.0);
		entityReal.set(1, 0, 0.0);
		entityImag.set(1, 0, 1.0);
		relationPhase.set(0, 0, Math.PI / 2.0);
		double score = RotatE.score(new Triple(0, 1, 0),
				entityReal, entityImag, relationPhase, 3.0);
		assertClose(3.0, score, 1e-12, "Un quart de tour doit mapper la tête sur la queue.");
	}

	private static void testGradientAgainstFiniteDifferences() throws Exception {
		Matrix entityReal = new Matrix(2, 1);
		Matrix entityImag = new Matrix(2, 1);
		Matrix relationPhase = new Matrix(1, 1);
		entityReal.set(0, 0, 0.71);
		entityImag.set(0, 0, -0.23);
		entityReal.set(1, 0, 0.31);
		entityImag.set(1, 0, 1.07);
		relationPhase.set(0, 0, 0.41);
		Triple triple = new Triple(0, 1, 0);
		double lambda = 0.037;
		double margin = 1.7;

		Matrix realGradient = new Matrix(2, 1);
		Matrix imagGradient = new Matrix(2, 1);
		Matrix phaseGradient = new Matrix(1, 1);
		new StochasticGradient(triple, 1.0, entityReal, entityImag, relationPhase,
				realGradient, imagGradient, phaseGradient, lambda, 1.0, margin)
				.calculateGradient();

		assertClose(numericalGradient(entityReal, entityImag, relationPhase,
				triple, lambda, margin, 0), realGradient.get(0, 0), 1e-5, "Gradient entity réel tête incorrect.");
		assertClose(numericalGradient(entityReal, entityImag, relationPhase,
				triple, lambda, margin, 1), imagGradient.get(0, 0), 1e-5, "Gradient entity imaginaire tête incorrect.");
		assertClose(numericalGradient(entityReal, entityImag, relationPhase,
				triple, lambda, margin, 2), realGradient.get(1, 0), 1e-5, "Gradient entity réel queue incorrect.");
		assertClose(numericalGradient(entityReal, entityImag, relationPhase,
				triple, lambda, margin, 3), imagGradient.get(1, 0), 1e-5, "Gradient entity imaginaire queue incorrect.");
		assertClose(numericalGradient(entityReal, entityImag, relationPhase,
				triple, lambda, margin, 4), phaseGradient.get(0, 0), 1e-5, "Gradient de phase de relation incorrect.");
	}

	private static double numericalGradient(Matrix entityReal, Matrix entityImag,
			Matrix relationPhase, Triple triple, double lambda, double margin,
			int parameter) throws Exception {
		double epsilon = 1e-6;
		double original = getParameter(entityReal, entityImag, relationPhase, parameter);
		setParameter(entityReal, entityImag, relationPhase, parameter, original + epsilon);
		double positiveLoss = loss(entityReal, entityImag, relationPhase, triple, lambda, margin);
		setParameter(entityReal, entityImag, relationPhase, parameter, original - epsilon);
		double negativeLoss = loss(entityReal, entityImag, relationPhase, triple, lambda, margin);
		setParameter(entityReal, entityImag, relationPhase, parameter, original);
		return (positiveLoss - negativeLoss) / (2.0 * epsilon);
	}

	private static double getParameter(Matrix real, Matrix imag, Matrix phase,
			int parameter) throws Exception {
		if (parameter == 0) return real.get(0, 0);
		if (parameter == 1) return imag.get(0, 0);
		if (parameter == 2) return real.get(1, 0);
		if (parameter == 3) return imag.get(1, 0);
		return phase.get(0, 0);
	}

	private static void setParameter(Matrix real, Matrix imag, Matrix phase,
			int parameter, double value) throws Exception {
		if (parameter == 0) real.set(0, 0, value);
		else if (parameter == 1) imag.set(0, 0, value);
		else if (parameter == 2) real.set(1, 0, value);
		else if (parameter == 3) imag.set(1, 0, value);
		else phase.set(0, 0, value);
	}

	private static double loss(Matrix real, Matrix imag, Matrix phase,
			Triple triple, double lambda, double margin) throws Exception {
		double score = RotatE.score(triple, real, imag, phase, margin);
		double regularization = lambda * (real.get(0, 0) * real.get(0, 0)
				+ imag.get(0, 0) * imag.get(0, 0)
				+ real.get(1, 0) * real.get(1, 0)
				+ imag.get(1, 0) * imag.get(1, 0));
		return Math.log1p(Math.exp(score)) - score + regularization;
	}

	private static void assertClose(double expected, double actual,
			double tolerance, String message) {
		if (Math.abs(expected - actual) > tolerance) {
			throw new AssertionError(message + " attendu=" + expected + ", obtenu=" + actual);
		}
	}
}
