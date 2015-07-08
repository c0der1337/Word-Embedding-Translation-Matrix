import java.util.HashMap;

import org.nd4j.linalg.api.ndarray.INDArray;

public class WordEmbeddingSpace {
	private INDArray matrix;
	private HashMap<Integer, Integer> row2id;
	
	
 public WordEmbeddingSpace(){
	 createRow2Id();
	 //if lexicon is provided, only data occurring in the lexicon is loaded
	 
	 
 }
 private void createRow2Id(){
	 
 }

}
