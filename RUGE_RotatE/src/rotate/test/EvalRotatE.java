package rotate.test;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashMap;

import model.struct.Matrix;
import model.struct.Triple;
import rotate.train.RotatE;

public class EvalRotatE {
	private final int numberOfEntities;
	private final double margin;
	private final Matrix entityReal;
	private final Matrix entityImag;
	private final Matrix relationPhase;
	private final HashMap<String, Boolean> knownTriples = new HashMap<String, Boolean>();

	private EvalRotatE(int entities, int relations, int dimensions, double margin,
			String entityRealPath, String entityImagPath, String relationPhasePath)
			throws Exception {
		numberOfEntities = entities;
		this.margin = margin;
		entityReal = new Matrix(entities, dimensions);
		entityImag = new Matrix(entities, dimensions);
		relationPhase = new Matrix(relations, dimensions);
		entityReal.load(entityRealPath);
		entityImag.load(entityImagPath);
		relationPhase.load(relationPhasePath);
	}

	private void loadKnownTriples(String... paths) throws Exception {
		for (String path : paths) {
			try (BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(path), "UTF-8"))) {
				String line;
				while ((line = reader.readLine()) != null) {
					knownTriples.put(line.trim(), true);
				}
			}
		}
	}

	private void evaluate(String testPath) throws Exception {
		double filteredReciprocalRanks = 0.0;
		double rawReciprocalRanks = 0.0;
		int filteredHitsAt10 = 0;
		int rawHitsAt10 = 0;
		int rankCount = 0;

		try (BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(testPath), "UTF-8"))) {
			String line;
			while ((line = reader.readLine()) != null) {
				String[] fields = line.trim().split("\\s+");
				if (fields.length != 3) {
					throw new IllegalArgumentException("Triplet de test mal formé : " + line);
				}
				int head = Integer.parseInt(fields[0]);
				int relation = Integer.parseInt(fields[1]);
				int tail = Integer.parseInt(fields[2]);
				Triple target = new Triple(head, tail, relation);
				double targetScore = RotatE.score(
						target, entityReal, entityImag, relationPhase, margin);

				int rawHeadRank = 1;
				int filteredHeadRank = 1;
				int rawTailRank = 1;
				int filteredTailRank = 1;
				for (int candidate = 0; candidate < numberOfEntities; candidate++) {
					double headScore = RotatE.score(new Triple(candidate, tail, relation),
							entityReal, entityImag, relationPhase, margin);
					if (headScore > targetScore) {
						rawHeadRank++;
						if (!knownTriples.containsKey(candidate + "\t" + relation + "\t" + tail)) {
							filteredHeadRank++;
						}
					}
					double tailScore = RotatE.score(new Triple(head, candidate, relation),
							entityReal, entityImag, relationPhase, margin);
					if (tailScore > targetScore) {
						rawTailRank++;
						if (!knownTriples.containsKey(head + "\t" + relation + "\t" + candidate)) {
							filteredTailRank++;
						}
					}
				}

				rawReciprocalRanks += 1.0 / rawHeadRank + 1.0 / rawTailRank;
				filteredReciprocalRanks += 1.0 / filteredHeadRank + 1.0 / filteredTailRank;
				if (rawHeadRank <= 10) rawHitsAt10++;
				if (rawTailRank <= 10) rawHitsAt10++;
				if (filteredHeadRank <= 10) filteredHitsAt10++;
				if (filteredTailRank <= 10) filteredHitsAt10++;
				rankCount += 2;
			}
		}
		if (rankCount == 0) {
			throw new IllegalStateException("Aucun triplet de test à évaluer.");
		}
		System.out.println("Filtered MRR: " + filteredReciprocalRanks / rankCount);
		System.out.println("Filtered Hits@10: " + (double) filteredHitsAt10 / rankCount);
		System.out.println("Raw MRR: " + rawReciprocalRanks / rankCount);
		System.out.println("Raw Hits@10: " + (double) rawHitsAt10 / rankCount);
	}

	public static void main(String[] args) throws Exception {
		if (args.length != 9 && args.length != 10) {
			throw new IllegalArgumentException("Usage: EvalRotatE <entities> <relations> <dimensions> "
					+ "<entity-real> <entity-imag> <relation-phase> <train> <valid> <test> [margin]");
		}
		int entities = Integer.parseInt(args[0]);
		int relations = Integer.parseInt(args[1]);
		int dimensions = Integer.parseInt(args[2]);
		double margin = args.length == 10 ? Double.parseDouble(args[9]) : 12.0;
		EvalRotatE evaluator = new EvalRotatE(entities, relations, dimensions, margin,
				args[3], args[4], args[5]);
		evaluator.loadKnownTriples(args[6], args[7], args[8]);
		evaluator.evaluate(args[8]);
	}
}
