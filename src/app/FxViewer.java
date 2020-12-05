package app;

import app.view.GlobeViewer;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ToolBar;
import javafx.stage.Stage;

public class FxViewer extends Application{
	public static void main(String[] args) {
		launch();
	}
	
	GlobeData dummy = new GlobeData(20, 30, 10).random();
	GlobeViewer gv = new GlobeViewer(dummy);
	Button timeStep = new Button("Step");
	ToolBar simulationControls = new ToolBar(timeStep);
	
	
	@Override
	public void start(Stage stage) throws Exception {
		stage.setScene(new Scene(gv, 900, 600));
		gv.setBottom(simulationControls);
		stage.show();
	}
}
