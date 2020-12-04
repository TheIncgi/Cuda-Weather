package app.view;

import javafx.scene.control.Label;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;

import static app.MathUtils.clamp;
import static app.MathUtils.map;

public class FloatVisulaizationTile extends Pane {
	boolean asColor = false;
	Rectangle rect = new Rectangle();
	Label label = new Label();
	public FloatVisulaizationTile() {
		rect.setWidth(new Label("100%").getWidth());
		rect.setHeight(rect.getWidth());
		rect.setFill(Color.TRANSPARENT);
		this.getChildren().addAll(rect, label);
	}
	
	public void setTempColor(float x) {
		// 0d h181
		//70d h126
		//105 h0
		label.setText(""+(int)x);
		x = clamp(x, 0, 105);
		if(x<=70)
			x = map(x, 0, 70, 181, 126);
		else
			x = map(x, 70, 105, 126, 0);
		rect.setFill(Color.RED.deriveColor(x, 1, 1, 1));
	}
	
	public void setPercentColor(float x) {
		// 0d h181
		//70d h126
		//105 h0
		label.setText(""+((int)(x*100))+"%");
		x =  clamp(x, 0, 1);
		x *= 181;
		rect.setFill(Color.RED.deriveColor(x, 1, 1, 1));
	}
	
	
}
