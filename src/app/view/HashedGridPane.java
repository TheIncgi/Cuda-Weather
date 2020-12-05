package app.view;

import java.util.HashMap;

import javafx.collections.ListChangeListener;
import javafx.scene.Node;
import javafx.scene.layout.GridPane;

public class HashedGridPane extends GridPane{
	private HashMap<Integer, HashMap<Integer, Node>> stuff = new HashMap<>();
	private String debug;
	
	public HashedGridPane() {
	}
	
	public HashedGridPane(String debug) {
		this.debug = debug;
	}
	
	@Override
	public String toString() {
		return this.debug==null?super.toString() : "HashedGridPane: "+debug;
	}
	
	//TODO handle removal
//	public HashedGridPane() {
//		super();
//		getChildren().removeListener(new ListChangeListener<Node>() {
//			@Override
//			public void onChanged(Change<? extends Node> c) {
//				
//			}
//		});
//	}

	public Node getNode(int x, int y) {
		HashMap<Integer, Node> z = stuff.get(x);
		if(z==null) return null;
		return z.get(y);
	}
	
	@Override
	public void add(Node child, int columnIndex, int rowIndex) {
		stuff.computeIfAbsent(columnIndex, k->new HashMap<>())
		.put(rowIndex, child);
		super.add(child, columnIndex, rowIndex);
	}
	
	@Override
	public void add(Node child, int columnIndex, int rowIndex, int colspan, int rowspan) {
		stuff.computeIfAbsent(columnIndex, k->new HashMap<>())
		.put(rowIndex, child);
		super.add(child, columnIndex, rowIndex, colspan, rowspan);
	}
	
	@Override
	public void addColumn(int columnIndex, Node... children) {
		HashMap<Integer, Node> x = stuff.computeIfAbsent(columnIndex, k->new HashMap<>());
		for(int rowIndex = 0; rowIndex<children.length; rowIndex++) 
			x.put(rowIndex, children[rowIndex]);
		super.addColumn(columnIndex, children);
	}
	
	@Override
	public void addRow(int rowIndex, Node... children) {
		for(int i = 0; i<children.length; i++)
		stuff.computeIfAbsent(i, k->new HashMap<>())
		.put(i, children[i]);
		super.addRow(rowIndex, children);
	}
	
	
}
