
public class TestingPair{
	//Class for handling a training example
	String source;
	String target;
	int id;
	public TestingPair(String _source, String _target){
		source = _source;
		target = _target;
	}
	public TestingPair(String _source, String _target, int _id){
		source = _source;
		target = _target;
		id = _id;
	}
}
