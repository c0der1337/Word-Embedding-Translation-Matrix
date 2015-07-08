import java.awt.geom.FlatteningPathIterator;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;


public class Train_TM {
	//static String data_path = "C://Users//Patrick//Documents//master arbeit//original_code//data//Wordnet//";
	//Data
	static String trainData_path = "C://Users//Patrick//Documents//master arbeit//Translation Matrix//data//";
	static String we_Src_file = "EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt";
	static String we_Tar_file = "IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt";
	static String trn_data_file = "OPUS_en_it_europarl_train_5K.txt";
	static String test_data_file = "OPUS_en_it_europarl_test.txt";
	
	static private int we_dimension = 300; // dimension of word embeddings
	static private int we_vocabsize = 50000; // number of words
	
	//
	static boolean train_or_load_tm = true; //true = train, load = false
	static boolean reduce_target_space = true;
	static int target_space_reduction_to = 50000;
	
	//Training data:
	private static ArrayList<TrainingPair> trainingDataSrcTar = new ArrayList();
	private static ArrayList<String> traingWordList_Source= new ArrayList();
	private static ArrayList<String> traingWordList_Target= new ArrayList();
	//Test data:
	private static ArrayList<TrainingPair> testDataSrcTar = new ArrayList();
	private static ArrayList<String> testWordList_Source= new ArrayList();
	private static ArrayList<String> testWordList_Target= new ArrayList();
	private static HashMap<String, Integer> testWordList_WordId= new HashMap<String, Integer>();
	//Source Training data
	private static HashMap<Integer, String> trnVocabNumWord_Src = new HashMap<Integer, String>(); //contains vocab of word vectors and index of a word
	private static HashMap<String, Integer> trnVocabWordNum_Src = new HashMap<String, Integer>();
	private static HashMap<Integer, INDArray> trnWorvectorNumVec_Src = new HashMap<Integer, INDArray>(); // return word vector for a given word index
	private static INDArray trnWordVectorMaxtrixLoaded_Source; 
	//Target Training data
	private static HashMap<Integer, String> trnVocabNumWord_Tar = new HashMap<Integer, String>(); //contains vocab of word vectors and index of a word
	private static HashMap<String, Integer> trnVocabWordNum_Tar = new HashMap<String, Integer>();
	private static HashMap<Integer, INDArray> trnWorvectorNumVec_Tar = new HashMap<Integer, INDArray>(); // return word vector for a given word index
	private static INDArray trnWordVectorMaxtrixLoaded_Tar; 
	//Source Test data
	private static HashMap<Integer, String> testVocabNumWord_Src = new HashMap<Integer, String>(); //contains vocab of word vectors and index of a word
	private static HashMap<String, Integer> testVocabWordNum_Src = new HashMap<String, Integer>();
	private static HashMap<Integer, INDArray> testWorvectorNumVec_Src = new HashMap<Integer, INDArray>(); // return word vector for a given word index
	private static INDArray testWordVectorMaxtrixLoaded_Source;  //
	//Target Test data
	private static HashMap<Integer, String> testVocabNumWord_Tar = new HashMap<Integer, String>(); //contains vocab of word vectors and index of a word
	private static HashMap<String, Integer> testVocabWordNum_Tar = new HashMap<String, Integer>();
	private static HashMap<Integer, INDArray> testWorvectorNumVec_Tar = new HashMap<Integer, INDArray>(); // return word vector for a given word index
	private static INDArray testWordVectorMaxtrixLoaded_Tar; 

	public static void main(String[] args) throws IOException {
		try {
			trainData_path = args[0];
			we_Src_file = args[2];
			we_Tar_file = args[3];
			trn_data_file = args[4];
			test_data_file = args[5];
			we_dimension = Integer.parseInt(args[6]);
			we_vocabsize = Integer.parseInt(args[7]);
		} catch (Exception e) {
			trainData_path = "C://Users//Patrick//Documents//master arbeit//Translation Matrix//data//";
			we_Src_file = "EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt";
			we_Tar_file = "IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt";
			trn_data_file = "OPUS_en_it_europarl_train_5K.txt";
			test_data_file = "OPUS_en_it_europarl_test.txt";
			we_dimension = 300; // dimension of word embeddings
			we_vocabsize = 200000; // number of words
		}
		//Initialize translation matrix
		INDArray tm= Nd4j.create(we_dimension,we_dimension);;
		
		if (train_or_load_tm==true) {
			// Train a translation matrix tm 
			tm = train();
		}else{
			//Load translation matrix
			tm = Nd4j.readTxt(trainData_path+"//translationmatrix.txt", ",");
		}
		
		//Test
		test(tm);

		/*
		INDArray x2 = Nd4j.rand(3,3);
		System.out.println(x2);
		System.out.println(Nd4j.sortWithIndices(x2, 0, true)[0]);
		System.out.println(Nd4j.sortWithIndices(x2, 0, false)[0]);
		*/
	}
	
	static public void loadTrainingData(String trainData_path) throws IOException{
		FileReader fr = new FileReader(trainData_path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int TranCounter = 0;
	    while (line != null) {
	    	// System.out.println("line: "+line.split(" ")[0]+"|"+line.split(" ")[1]);
	    	trainingDataSrcTar.add(new TrainingPair(line.split(" ")[0], line.split(" ")[1]));
	    	traingWordList_Source.add(line.split(" ")[0]);
	    	traingWordList_Target.add(line.split(" ")[1]);
	    	line = br.readLine();
	    	TranCounter++;
		}   
	    br.close();
	    
	    //number of entities need increased by one to handle the zero entry
	    System.out.println(TranCounter + " EN-IT word translation for training loaded");
	}
	
	static public void loadWordVectors_SourceTraining(String trainData_path, ArrayList<String> sourceWords2load4Training) throws IOException{
		// vectors in source language. Space-separated, with string 
        // identifier as first column (dim+1 columns, where dim is the dimensionality
        // of the space)
		
		FileReader fr = new FileReader(trainData_path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int weCounter = 0;
	    //Skip first line, because it contains only information about the encoding
	    line = br.readLine();

	    while (line != null) {
	    	String wv[] = line.split(" ");
	    	//Only load word vectors for words from the training data
		    if (sourceWords2load4Training.contains(wv[0])) {
		    	trnVocabNumWord_Src.put(weCounter,  wv[0]);
		    	trnVocabWordNum_Src.put(wv[0], weCounter);
		    	INDArray wordvector = Nd4j.create(we_dimension);
		    	for (int i = 1; i < wv.length; i++) {
		    		wordvector.putScalar(i-1, Double.parseDouble(wv[i]));
				}
		    	trnWorvectorNumVec_Src.put(weCounter, wordvector);
		    	if (weCounter<=10) {
					System.out.println(weCounter+": sourceWordTwoLoad: "+wv[0]+" |wordvector:"+wordvector);
				}
		    	weCounter++;
			}	
	    	line = br.readLine();
	    		
		} 
	    trnWordVectorMaxtrixLoaded_Source=  Nd4j.create(trnVocabNumWord_Src.size(),we_dimension);
	    for (int i = 0; i < trnWorvectorNumVec_Src.size(); i++) {
	    	if(i<3){
	    		System.out.println(i+": "+trnWorvectorNumVec_Src.get(i));
	    	}
	    	trnWordVectorMaxtrixLoaded_Source.putRow(i, trnWorvectorNumVec_Src.get(i));
		}
	    br.close();
	    
	    //number of entities need increased by one to handle the zero entry
	    System.out.println(trnWorvectorNumVec_Src.size() + " Source word vectors for training loaded");
	}

	static public void loadWordVectors_TargetTraining(String trainData_path, ArrayList<String> targetWords2load4Training) throws IOException{
		// vectors in target language. Space-separated, with string 
        // identifier as first column (dim+1 columns, where dim is the dimensionality
        // of the space)
		
		FileReader fr = new FileReader(trainData_path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int weCounter = 0;
	    //Skip first line, because it contains only information about the encoding
	    line = br.readLine();

	    while (line != null) {
	    	String wv[] = line.split(" ");
	    	//Only load word vectors for words from the training data
	    	//if (wv[0].equals("are")) {
				//System.out.println("are: "+weCounter+" "+trainingDataSrcTar.get(weCounter).target);
			//}
		    if (traingWordList_Target.contains(wv[0])) {
		    	/*if (wv[0].equals("are")) {
					System.out.println("are: "+weCounter);
				}*/
		    	trnVocabNumWord_Tar.put(weCounter,  wv[0]);
		    	trnVocabWordNum_Tar.put(wv[0], weCounter);
		    	INDArray wordvector = Nd4j.create(we_dimension);
		    	for (int i = 1; i < wv.length; i++) {
		    		wordvector.putScalar(i-1, Double.parseDouble(wv[i]));
				}
		    	trnWorvectorNumVec_Tar.put(weCounter, wordvector);
		    	weCounter++;
			}	
	    	line = br.readLine();
	    		
		} 
	    trnWordVectorMaxtrixLoaded_Tar=  Nd4j.create(trnVocabNumWord_Tar.size(),we_dimension);
	    for (int i = 0; i < trnWorvectorNumVec_Tar.size(); i++) {
	    	trnWordVectorMaxtrixLoaded_Tar.putRow(i, trnWorvectorNumVec_Tar.get(i));
		}
	    br.close();
	    
	    //number of entities need increased by one to handle the zero entry
	    System.out.println(weCounter + " IT word vectors loaded");
	}
	
	static public void loadTestData(String testData_path) throws IOException{
		FileReader fr = new FileReader(testData_path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int TranCounter = 0;
	    int TranCounterAll = 0;
	    while (line != null) {
	    	TranCounterAll++;
	    	if (TranCounter <5) {
	    		//System.out.println("loadTestData: "+line.split(" ")[0]+"|"+line.split(" ")[1]);
			} 
	    	// Check if each word is contained in the source and target word spaces
			if (testVocabWordNum_Tar.containsKey(line.split(" ")[0])) {
		    	if (testWordList_Source.contains(line.split(" ")[0]) == false) {
		    		// new translation pair
			    	testDataSrcTar.add(new TrainingPair(line.split(" ")[0], line.split(" ")[1]));
			    	testWordList_WordId.put(line.split(" ")[0], TranCounter);
			    	testWordList_Source.add(line.split(" ")[0]);
			    	testWordList_Target.add(line.split(" ")[1]);
			    	TranCounter++;
				}else{
					// add additional translation posibility for the source word
		    		testDataSrcTar.get(testWordList_WordId.get(line.split(" ")[0])).addTarget(line.split(" ")[1]);;
					
				}
			}
	    	line = br.readLine();
	    	
		}   
	    br.close();
	    
	    //number of entities need increased by one to handle the zero entry
	    System.out.println(TranCounter + " EN-IT word translation to test accuracy loaded");
	}
	
	static public void loadWordVectors_SourceTest(String trainData_path, ArrayList<String> sourceWords2load4Test) throws IOException{
		// vectors in source language. Space-separated, with string 
        // identifier as first column (dim+1 columns, where dim is the dimensionality
        // of the space)
		
		FileReader fr = new FileReader(trainData_path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    int weCounter = 0;
	    //Skip first line, because it contains only information about the encoding
	    line = br.readLine();
	    
	    while (line != null) {
	    	String wv[] = line.split(" ");
	    	//Only load word vectors for words from the training data
		    if (sourceWords2load4Test.contains(wv[0])) {
		    	if (weCounter<=5) {
					//System.out.println(" "+weCounter+" :"+wv[0]+">"+testDataSrcTar.get(weCounter).source);
				}
		    	testVocabNumWord_Src.put(weCounter,  wv[0]);
		    	testVocabWordNum_Src.put(wv[0], weCounter);
		    	INDArray wordvector = Nd4j.create(we_dimension);
		    	for (int i = 1; i < wv.length; i++) {
		    		wordvector.putScalar(i-1, Double.parseDouble(wv[i]));
				}
		    	testWorvectorNumVec_Src.put(weCounter, wordvector);
		    	weCounter++;
			}	
	    	line = br.readLine();
	    		
		} 
	    testWordVectorMaxtrixLoaded_Source=  Nd4j.create(testVocabNumWord_Src.size(),we_dimension);
	    for (int i = 0; i < testWorvectorNumVec_Src.size(); i++) {
	    	testWordVectorMaxtrixLoaded_Source.putRow(i, testWorvectorNumVec_Src.get(i));
		}
	    br.close();
	    
	    //number of entities need increased by one to handle the zero entry
	    System.out.println(testWorvectorNumVec_Src.size() + " Source word vectors for test loaded");
	}
	
	static public void loadAllWordVectors_TargetTest(String trainData_path) throws IOException{
		// vectors in source language. Space-separated, with string 
        // identifier as first column (dim+1 columns, where dim is the dimensionality
        // of the space)
		
		FileReader fr = new FileReader(trainData_path);
	    BufferedReader br = new BufferedReader(fr);
	    String line = br.readLine();
	    System.out.println("Target space:" +line);
	    int weCounter = 0;
	    //Skip first line, because it contains only information about the encoding
	    line = br.readLine();
	    if (reduce_target_space ==false) {
	    	testWordVectorMaxtrixLoaded_Tar = Nd4j.create(we_vocabsize,we_dimension);
	    	target_space_reduction_to = we_vocabsize;
		}else{
			testWordVectorMaxtrixLoaded_Tar = Nd4j.create(target_space_reduction_to,we_dimension);
		}

	    while (line != null & weCounter<target_space_reduction_to) {
	    //while (line != null) {
	    	String wv[] = line.split(" ");
	    	//System.out.println(weCounter + ": "+wv[0]);
	    	testVocabNumWord_Tar.put(weCounter,  wv[0]);
	    	testVocabWordNum_Tar.put(wv[0], weCounter);
	    	INDArray wordvector = Nd4j.create(we_dimension);
	    	for (int i = 1; i < wv.length; i++) {
	    		wordvector.putScalar(i-1, Double.parseDouble(wv[i]));
			}
	    	testWorvectorNumVec_Tar.put(weCounter, wordvector);
	    	testWordVectorMaxtrixLoaded_Tar.putRow(weCounter, wordvector);
	    	line = br.readLine();
	    	weCounter++;	
		}   
	    br.close();
	    
	    //number of entities need increased by one to handle the zero entry
	    System.out.println(weCounter + " IT word vectors for test loaded");
	}
	
	
	static public INDArray normalizeWordVectorsSpace(INDArray WordVectorMatrix){
				INDArray row_norms = sqrt(Nd4j.sum(WordVectorMatrix.mul(WordVectorMatrix),1));
				//System.out.println("row norms: "+row_norms);
				row_norms = Nd4j.toFlattened(Nd4j.ones(row_norms.columns()).div(row_norms));
				//System.out.println("row norms: "+row_norms);
				//System.out.println("ROW: "+WordVectorMatrix.mulRowVector(row_norms));
				//System.out.println("Column: "+WordVectorMatrix.mulColumnVector(row_norms));
				//System.out.println("MMUL: "+WordVectorMatrix.mmul(row_norms));
				return WordVectorMatrix.mulColumnVector(row_norms);
	}

	static public INDArray train(){
		System.out.println("+++ Train Translation Matrix");
		
		//load training data
		try {
			loadTrainingData(trainData_path + trn_data_file);
			loadWordVectors_SourceTraining(trainData_path + we_Src_file, traingWordList_Source);
			loadWordVectors_TargetTraining(trainData_path + we_Tar_file, traingWordList_Target);
		} catch (IOException e1) {
			System.out.println("Error while loading training data.");
			e1.printStackTrace();
		}
		//TODO Check if for each training pair vectors are found
		
		System.out.println("Training data loaded. Source-Target Training Pairs: "+trainingDataSrcTar.size()+" Size of Source Matrix: "+trnWordVectorMaxtrixLoaded_Source.columns()+" | " + trnWordVectorMaxtrixLoaded_Source.rows()+" Size of Target Matrix: "+trnWordVectorMaxtrixLoaded_Tar.columns()+" | " + trnWordVectorMaxtrixLoaded_Tar.rows());
		
		// normalize word vectors spaces
		trnWordVectorMaxtrixLoaded_Source = normalizeWordVectorsSpace(trnWordVectorMaxtrixLoaded_Source);			
		trnWordVectorMaxtrixLoaded_Tar = normalizeWordVectorsSpace(trnWordVectorMaxtrixLoaded_Tar);
			
		// get valid data for training (only pairs with word vectors for both source and target (not correct implemented))
		int usableWordPairs = 0;
		INDArray m1 = Nd4j.create(trainingDataSrcTar.size(),we_dimension);
		INDArray m2 = Nd4j.create(trainingDataSrcTar.size(),we_dimension);
		
		for (int i = 0; i < trainingDataSrcTar.size(); i++) {
			m1.putRow(i, trnWordVectorMaxtrixLoaded_Source.getRow(trnVocabWordNum_Src.get(traingWordList_Source.get(i))));
			m2.putRow(i, trnWordVectorMaxtrixLoaded_Tar.getRow(trnVocabWordNum_Tar.get(traingWordList_Target.get(i))));
			usableWordPairs++;
			//System.out.println("Pair: " +usableWordPairs+" "+traingWordsEN.get(i) + " | " +  traingWordsIT.get(i));
		}
		System.out.println("UsableWordPairs for training: "+usableWordPairs);
		
		//generate double arrays for solver
		double[][] m1matrix = new double[trainingDataSrcTar.size()][we_dimension];
		for (int i = 0; i < trainingDataSrcTar.size(); i++) { 
			for (int j = 0; j < we_dimension; j++) {
				m1matrix[i][j] = m1.getDouble(i,j);
			}
		}
		double[][] m2matrix = new double[trainingDataSrcTar.size()][we_dimension];
		for (int i = 0; i < m2.rows(); i++) { 
			for (int j = 0; j < m2.columns(); j++) {
				m2matrix[i][j] = m2.getDouble(i,j);
			}
		}
		//Training: solving a linear system
		RealMatrix matrix = new Array2DRowRealMatrix(m1matrix,false);
		DecompositionSolver solver1 = new SingularValueDecomposition(matrix).getSolver();
		RealMatrix solution1 = solver1.solve(new Array2DRowRealMatrix(m2matrix,false));
		//System.out.println("solution matrix shape: "+solution1.getRowDimension()+"|"+solution1.getColumnDimension());
		
		//Convert solution matrix back into an INDArray
		INDArray tm = Nd4j.create(we_dimension,we_dimension);
		double [][] tmDouble = solution1.getData();
		for (int j = 0; j < we_dimension; j++) {
			for (int i = 0; i < we_dimension; i++) {
				tm.put(j, i,tmDouble[j][i]);
			}
		}
		
		//Save translation matrix
		try {
			Nd4j.writeTxt( tm, trainData_path+"//translationmatrix.txt", ",");
		} catch (IOException e) {
			System.out.println("Error, cant save translationmatrix.txt");
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return tm;
	}
	static public void test(INDArray tm){
		System.out.println("+++ Testing the translations");
		
		//Load test data
		try {
			// ALL words in the _target_space are used as the search space (example: 200k words).	
			loadAllWordVectors_TargetTest(trainData_path + we_Tar_file);
			// we only load training examples were a word vector is found for the gold key
			loadTestData(trainData_path + test_data_file);
			// we only need to load vectors for the words in test into the source space (example: 1869 words).
			loadWordVectors_SourceTest(trainData_path + we_Src_file, testWordList_Source);
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Error while loading test data.");
			e.printStackTrace();
		}
		
		System.out.println("Training data loaded. Source-Target Training Pairs: "+trainingDataSrcTar.size()+" Size of Source Matrix: "+testWordVectorMaxtrixLoaded_Source.columns()+" | " + testWordVectorMaxtrixLoaded_Source.rows()+" Size of Target Matrix: "+testWordVectorMaxtrixLoaded_Tar.columns()+" | " + testWordVectorMaxtrixLoaded_Tar.rows());
		
		//Normalize both spaces:
		testWordVectorMaxtrixLoaded_Source = normalizeWordVectorsSpace(testWordVectorMaxtrixLoaded_Source);
		testWordVectorMaxtrixLoaded_Tar = normalizeWordVectorsSpace(testWordVectorMaxtrixLoaded_Tar);
		
		//Applying the translation matrix on the source word embeddings in order to map/transform the source vectors into the target vector space
		INDArray mapped_source_sp  = testWordVectorMaxtrixLoaded_Source.mmul(tm);
		
		
		
		//Retrieving translations
		
		//Normalize mapped_source space
		mapped_source_sp = normalizeWordVectorsSpace(mapped_source_sp);
		
		// Computing cosines to see which vectors of the target space are most similar / nearest to the source vectors
		INDArray sim_mat = testWordVectorMaxtrixLoaded_Tar.neg().mmul(mapped_source_sp.transpose());
		//Nd4j.writeTxt( tm, trainData_path+"//sim_mat_"+sim_mat.shape()[0]+"|"+sim_mat.shape()[1]+".txt", ",");
		
		// Sorting target space elements to get the most similar at top
		INDArray srtd_idx = Nd4j.sortWithIndices(sim_mat, 0, true)[0];
		//System.out.println(srtd_idx);
		
		System.out.println("Test data example: Source Word 0: "+testWordList_Source.get(0)+" | target Word 0"+ testDataSrcTar.get(0).getTargets());
		
		// Evaluation of the test results
		int rankcounter=0; // sum of the best correct rank of all test examples (if no correct translation is found: k+1 added)
		double firstrank = 0; // count number of first rank correct translations of all test examples
		double firstXrank = 0; // count number of first x rank correct translations of all test examples
		int bestrank = 0; // 
		int numOfTry = 5;
		//INDArray ranks = Nd4j.zeros(testDataSrcTar.size());
		for (int i = 0; i < testDataSrcTar.size(); i++) {
			boolean translated = false;
			// Get the index of the source word to translate
			int source_idx = testVocabWordNum_Src.get(testWordList_Source.get(i));
			System.out.println("id: " +i +" | translation proposals for \"" + testWordList_Source.get(i)+"\" are: ");
			for (int k = 0; k < numOfTry; k++) {
				// Get the kth most similar / nearest word index / position in target space
				int target_idx = srtd_idx.getInt(k,source_idx);
				// Retrieve translation / word from the target space
				String translation = testVocabNumWord_Tar.get(target_idx);
				// Retrieve Score of simililarity
				double score = sim_mat.neg().getDouble((int)target_idx,source_idx);
				System.out.println(k+". translation: "+translation + " score: "+score);
				
				// Accuracy evaluation: is this translation correct?
				for (int j = 0; j < testDataSrcTar.get(i).getTargets().size(); j++) {
					if (translation.equals(testDataSrcTar.get(i).getTargets().get(j))) {
						if (translated==false) {
							rankcounter = rankcounter + k;
							bestrank = k;
							if (k==0) {
								firstrank++;
								firstXrank++;
							}else{
								firstXrank++;
							}
						}
						translated = true;
					}
				}
				if (translated==false) {
					rankcounter = rankcounter + k+1;
				}
			}
			if (translated==true) {
				System.out.println("Correct translation / gold key: "+ testDataSrcTar.get(i).getTargets() + "rank: " + bestrank);
			}else{
				System.out.println("Correct translation / gold key: "+ testDataSrcTar.get(i).getTargets() + "rank: NOT FOUND!");
			}
			System.out.println();
		}
		//Print out results
		System.out.println("Test data size: "+testVocabWordNum_Src.size());
		System.out.println("Number of correct translations (first rank): " +firstrank+" P1: " + (firstrank/testVocabWordNum_Src.size()));
		System.out.println("Number of correct translations (in first k ranks): " +firstXrank+" P"+numOfTry+": " + (firstXrank/testVocabWordNum_Src.size()));
		System.out.println("Average rank, if found: " + (rankcounter/testVocabWordNum_Src.size()));
		
	}
}



