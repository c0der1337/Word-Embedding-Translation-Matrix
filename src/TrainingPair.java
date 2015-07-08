import java.util.ArrayList;


public class TrainingPair{
	//Class for handling a training example
	private String source;
	private String target;
	private ArrayList<String> targets = new ArrayList();
	private int id;
	public TrainingPair(String _source, String _target){
		source = _source;
		target = _target;
		targets.add(target);
	}
	public TrainingPair(String _source, String _target, int _id){
		source = _source;
		target = _target;
		targets.add(target);
		id = _id;
	}
	public ArrayList<String> getTargets() {
		return targets;
	}
	public void setTargets(ArrayList<String> targets) {
		this.targets = targets;
	}
	
	public void addTarget(String target) {
		targets.add(target);
	}
	
	
	
	
	
}
