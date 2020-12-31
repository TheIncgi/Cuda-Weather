package app.view;

import java.util.HashMap;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;

public class DebugPanel extends TableView<DebugPanel.Row> {
	public Row rows[];
	private final HashMap<String, Row> lookup = new HashMap<>();
	
	@SuppressWarnings("unchecked")
	public DebugPanel(String...infos) {
		rows = new Row[infos.length];
		for (int i = 0; i < rows.length; i++) {
			rows[i] = new Row();
			rows[i].label.setText(infos[i]);
			lookup.put(infos[i], rows[i]);
		}
		ObservableList<Row> obsRow = FXCollections.observableArrayList(rows);
		TableColumn<Row, String> attribCol = new TableColumn<>("Attribute");
		attribCol.setCellValueFactory(new PropertyValueFactory<>("label"));
		TableColumn<Row, String> valueCol = new TableColumn<>("Value");
		valueCol.setCellValueFactory(new PropertyValueFactory<>("value"));
		
		valueCol.prefWidthProperty().bind(this.widthProperty().subtract(attribCol.widthProperty()));
		
		setItems(obsRow);
		getColumns().addAll(attribCol, valueCol);
	}
	
	public void changeValue(String key, String value) {
		Row r = lookup.get(key);
		if(r==null)
			System.err.println("Missing key '"+key+"'");
		else
			r.value.setText(value);
	}
	
	
	public static class Row {
		Label label = new Label();
		Label value = new Label();
		public Label getLabel() {
			return label;
		}
		public void setLabel(Label label) {
			this.label = label;
		}
		public Label getValue() {
			return value;
		}
		public void setValue(Label value) {
			this.value = value;
		}
		
	}
}
