package rotate.train;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import model.struct.Matrix;
import model.struct.Rule;
import model.struct.RuleSet;
import model.struct.Triple;
import model.struct.TripleSet;
import model.util.NegativeTripleGeneration;

public class RUGEModel {
	public TripleSet TrainingTriples;
	public TripleSet ValidateTriples;
	public TripleSet Triples;
	public RuleSet GroundRules;
	public Matrix Entity_Real_MatrixE;
	public Matrix Entity_Imag_MatrixE;
	public Matrix Relation_Phase_MatrixR;
	public Matrix Real_MatrixEGradient;
	public Matrix Imag_MatrixEGradient;
	public Matrix Phase_MatrixRGradient;

	public int NumRelation;
	public int NumEntity;
	public int NumFactor = 200;
	public int NumMiniBatch = 100;
	public int NumNeg = 10;
	public double Lambda = 0.01;
	public double GammaE = 0.5;
	public double GammaR = 0.5;
	public double C = 0.01;
	public double Margin = 12.0;
	public int NumIteration = 1000;
	public int OutputIterSkip = 50;
	public int NumValidationTriples = 10;

	private final String matrixEPrefix = "MatrixE";
	private final String matrixRPrefix = "MatrixR";

	public void Initialization(String strNumRelation, String strNumEntity,
			String trainingPath, String validationPath, String testingPath,
			String groundRulesPath) throws Exception {
		NumRelation = Integer.parseInt(strNumRelation);
		NumEntity = Integer.parseInt(strNumEntity);
		if (NumEntity <= 0 || NumRelation <= 0 || NumFactor <= 0
				|| NumMiniBatch <= 0 || NumNeg < 2 || Margin <= 0.0
				|| Lambda < 0.0 || GammaE < 0.0 || GammaR < 0.0
				|| NumIteration < 0 || OutputIterSkip <= 0
				|| NumValidationTriples <= 0) {
			throw new IllegalArgumentException("One or more RotatE/RUGE parameters are invalid.");
		}

		System.out.println("\nChargement des triplets");
		TrainingTriples = new TripleSet(NumEntity, NumRelation);
		ValidateTriples = new TripleSet(NumEntity, NumRelation);
		Triples = new TripleSet();
		TrainingTriples.load(trainingPath);
		ValidateTriples.subload(validationPath);
		Triples.loadStr(trainingPath);
		Triples.loadStr(validationPath);
		Triples.loadStr(testingPath);
		if (TrainingTriples.triples() == 0) {
			throw new IllegalArgumentException("The training file contains no triples.");
		}

		System.out.println("Loading grounded rules");
		GroundRules = new RuleSet(NumEntity, NumRelation);
		GroundRules.load(groundRulesPath);

		System.out.println("Initializing RotatE embeddings");
		Entity_Real_MatrixE = new Matrix(NumEntity, NumFactor);
		Entity_Imag_MatrixE = new Matrix(NumEntity, NumFactor);
		Relation_Phase_MatrixR = new Matrix(NumRelation, NumFactor);
		initializeEmbeddings();

		Real_MatrixEGradient = new Matrix(NumEntity, NumFactor);
		Imag_MatrixEGradient = new Matrix(NumEntity, NumFactor);
		Phase_MatrixRGradient = new Matrix(NumRelation, NumFactor);
		System.out.println("Initialization complete.");
	}

	private void initializeEmbeddings() throws Exception {
		Random random = new Random(123);
		double embeddingRange = (Margin + 2.0) / NumFactor;
		for (int i = 0; i < NumEntity; i++) {
			for (int j = 0; j < NumFactor; j++) {
				Entity_Real_MatrixE.set(i, j, (random.nextDouble() * 2.0 - 1.0) * embeddingRange);
				Entity_Imag_MatrixE.set(i, j, (random.nextDouble() * 2.0 - 1.0) * embeddingRange);
			}
		}
		for (int i = 0; i < NumRelation; i++) {
			for (int j = 0; j < NumFactor; j++) {
				Relation_Phase_MatrixR.set(i, j, random.nextDouble() * 2.0 * Math.PI - Math.PI);
			}
		}
	}

	public void RotatE_Learn() throws Exception {
		try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream("result.log"), "UTF-8"))) {
			RotatEMetricMonitor initialMetrics = newMetricMonitor();
			System.out.println("Initial evaluation on "
					+ Math.min(ValidateTriples.triples(), NumValidationTriples)
					+ " validation triples...");
			initialMetrics.calculateMetrics();
			double bestMRR = initialMetrics.dMRR;
			double bestHits = initialMetrics.dHits;
			int bestIteration = 0;
			saveBest();
			System.out.println("Initial evaluation complete: MRR=" + bestMRR
					+ "\tHits@10=" + bestHits);
			writer.write("Itération initiale : MRR=" + bestMRR + "\tHits@10=" + bestHits + "\n");
			writer.flush();

			long startTime = System.currentTimeMillis();
			for (int iteration = 1; iteration <= NumIteration; iteration++) {
				System.out.println("Iteration " + iteration + "/" + NumIteration
						+ ": preparing mini-batches...");
				HashMap<Integer, ArrayList<Triple>> positives = tripleBatches();
				HashMap<Integer, ArrayList<Triple>> headNegatives = tripleBatches();
				HashMap<Integer, ArrayList<Triple>> tailNegatives = tripleBatches();
				HashMap<Integer, ArrayList<Rule>> rules = ruleBatches();

				TrainingTriples.randomShuffle();
				for (int index = 0; index < TrainingTriples.triples(); index++) {
					if (index > 0 && index % 100000 == 0) {
						System.out.println("  Triples prepared: " + index + "/"
								+ TrainingTriples.triples());
					}
					Triple positive = TrainingTriples.get(index);
					NegativeTripleGeneration generator = new NegativeTripleGeneration(
							positive, NumEntity, NumRelation);
					int negativesPerSide = NumNeg / 2;
					int batch = index % NumMiniBatch;
					HashSet<Triple> headSet = generator.generateHeadNegTriple(negativesPerSide);
					HashSet<Triple> tailSet = generator.generateTailNegTriple(negativesPerSide);
					positives.get(batch).add(positive);
					headNegatives.get(batch).addAll(headSet);
					tailNegatives.get(batch).addAll(tailSet);
				}

				GroundRules.randomShuffle();
				for (int index = 0; index < GroundRules.rules(); index++) {
					rules.get(index % NumMiniBatch).add(GroundRules.get(index));
				}

				for (int batch = 0; batch < NumMiniBatch; batch++) {
					new StochasticUpdate(positives.get(batch), headNegatives.get(batch),
							tailNegatives.get(batch), rules.get(batch),
							Entity_Real_MatrixE, Entity_Imag_MatrixE, Relation_Phase_MatrixR,
							Real_MatrixEGradient, Imag_MatrixEGradient, Phase_MatrixRGradient,
							GammaE, GammaR, Lambda, C, Margin).stochasticIteration();
					if ((batch + 1) % 10 == 0 || batch + 1 == NumMiniBatch) {
						System.out.println("  Mini-batches updated: " + (batch + 1)
								+ "/" + NumMiniBatch);
					}
				}
				System.out.println("Iteration " + iteration + "/" + NumIteration + " complete.");

				if (iteration % OutputIterSkip == 0) {
					System.out.println("Evaluating "
							+ Math.min(ValidateTriples.triples(), NumValidationTriples)
							+ " validation triples...");
					RotatEMetricMonitor metrics = newMetricMonitor();
					metrics.calculateMetrics();
					System.out.println("Evaluation complete: MRR=" + metrics.dMRR
							+ "\tHits@10=" + metrics.dHits);
					writer.write("Itération " + iteration + " : MRR=" + metrics.dMRR
							+ "\tHits@10=" + metrics.dHits + "\n");
					if (metrics.dMRR > bestMRR) {
						bestMRR = metrics.dMRR;
						bestHits = metrics.dHits;
						bestIteration = iteration;
						saveBest();
					}
					writer.write("Meilleure itération " + bestIteration + " : MRR="
							+ bestMRR + "\tHits@10=" + bestHits + "\n");
					writer.flush();
				}
			}
			System.out.println("Temps total : " + (System.currentTimeMillis() - startTime) + " ms");
		}
	}

	private HashMap<Integer, ArrayList<Triple>> tripleBatches() {
		HashMap<Integer, ArrayList<Triple>> batches = new HashMap<Integer, ArrayList<Triple>>();
		for (int i = 0; i < NumMiniBatch; i++) {
			batches.put(i, new ArrayList<Triple>());
		}
		return batches;
	}

	private HashMap<Integer, ArrayList<Rule>> ruleBatches() {
		HashMap<Integer, ArrayList<Rule>> batches = new HashMap<Integer, ArrayList<Rule>>();
		for (int i = 0; i < NumMiniBatch; i++) {
			batches.put(i, new ArrayList<Rule>());
		}
		return batches;
	}

	private RotatEMetricMonitor newMetricMonitor() {
		return new RotatEMetricMonitor(ValidateTriples, Triples.tripleSet(),
				Entity_Real_MatrixE, Entity_Imag_MatrixE, Relation_Phase_MatrixR,
				Margin, NumValidationTriples);
	}

	private void saveBest() throws Exception {
		Entity_Real_MatrixE.output(matrixEPrefix + ".real.best");
		Entity_Imag_MatrixE.output(matrixEPrefix + ".imag.best");
		Relation_Phase_MatrixR.output(matrixRPrefix + ".phase.best");
	}
}
