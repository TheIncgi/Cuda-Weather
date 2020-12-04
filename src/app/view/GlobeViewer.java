package app.view;

import app.GlobeData;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Slider;
import javafx.scene.control.ToolBar;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;

public class GlobeViewer extends BorderPane{
	ScrollPane scrolly = new ScrollPane();
	Pane layers = new Pane();
	
	GridPane terainGrid = new GridPane(), 
			tempGrid = new GridPane(), 
			humidityGrid = new GridPane(), 
			cloudCoverGrid = new GridPane(), 
			windGrid = new GridPane(), 
			snowGrid = new GridPane(), 
			groundMoistureGrid = new GridPane();
	
	ComboBox<String> colorOverlay, NumberOverlay;
	CheckBox showWindDir = new CheckBox("Show Wind");
	Slider altitudeSlider;
	ToolBar layerModes = new ToolBar();
	
	
	public GlobeViewer(GlobeData gd) {
		for(int la=0; la<gd.longitudeDivisions; la++)
			for(int lo=0; lo<gd.latitudeDivisions; lo++)
				;//TODO
	}
	
	
}
