package app.view;

import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;

public class LoadingPane extends Pane{
	
	ProgressIndicator indicator = new ProgressIndicator();
	Label status = new Label();
	
	public LoadingPane() {
		VBox box = new VBox(20, indicator, status);
		box.layoutXProperty().bind(widthProperty().divide(2).subtract(box.widthProperty().divide(2)));
		box.layoutYProperty().bind(heightProperty().divide(2).subtract(box.heightProperty().divide(2)));
		getChildren().add(box);
	}

	public void status(String step, float progress) {
		this.status .setText(step);
		indicator.setProgress(progress);
	}
	
	
}
