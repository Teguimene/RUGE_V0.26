package rotate.train;

import basic.util.Arguments;

public class Program_RUGE_RotatE {
	public static void main(String[] args) throws Exception {
		Arguments arguments = new Arguments(args);
		RUGEModel model = new RUGEModel();
		String training = required(arguments, "train");
		String validation = required(arguments, "valid");
		String testing = required(arguments, "test");
		String rules = required(arguments, "rule");
		String relationCount = required(arguments, "m");
		String entityCount = required(arguments, "n");

		if (arguments.getValue("k") != null) {
			model.NumFactor = Integer.parseInt(arguments.getValue("k"));
		}
		if (arguments.getValue("d") != null) {
			model.Lambda = Double.parseDouble(arguments.getValue("d"));
		}
		if (arguments.getValue("c") != null) {
			model.C = Double.parseDouble(arguments.getValue("c"));
		}
		if (arguments.getValue("ne") != null) {
			model.NumNeg = Integer.parseInt(arguments.getValue("ne"));
		}
		if (arguments.getValue("ge") != null) {
			model.GammaE = Double.parseDouble(arguments.getValue("ge"));
		}
		if (arguments.getValue("gr") != null) {
			model.GammaR = Double.parseDouble(arguments.getValue("gr"));
		}
		if (arguments.getValue("gamma") != null) {
			model.Margin = Double.parseDouble(arguments.getValue("gamma"));
		}
		if (arguments.getValue("#") != null) {
			model.NumIteration = Integer.parseInt(arguments.getValue("#"));
		}
		if (arguments.getValue("skip") != null) {
			model.OutputIterSkip = Integer.parseInt(arguments.getValue("skip"));
		}
		if (arguments.getValue("eval") != null) {
			model.NumValidationTriples = Integer.parseInt(arguments.getValue("eval"));
		}

		model.Initialization(relationCount, entityCount, training, validation, testing, rules);
		System.out.println("\nStarting RUGE training with RotatE");
		model.RotatE_Learn();
		System.out.println("Training complete.");
	}

	private static String required(Arguments arguments, String name) {
		String value = arguments.getValue(name);
		if (value == null || value.length() == 0) {
			throw new IllegalArgumentException("Missing required argument: -" + name);
		}
		return value;
	}
}
