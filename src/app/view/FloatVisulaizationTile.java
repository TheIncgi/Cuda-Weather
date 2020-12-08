package app.view;

import javafx.scene.control.Label;
import javafx.scene.effect.GaussianBlur;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;

import static app.util.MathUtils.clamp;
import static app.util.MathUtils.map;

import app.GlobeData;
import app.GlobeData.GroundType;
import app.util.MathUtils;
import static app.util.MathUtils.inRange;
public class FloatVisulaizationTile extends Pane {
	boolean asColor = false;
	Rectangle rect = new Rectangle();
	Label label = new Label();
	int x, y;
	public FloatVisulaizationTile(int x, int y) {
		this.x = x; this.y = y;
		rect.setWidth(32);
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
		if(x<50)
			x = map(x, 0, 70, 181, 160);
		else if(x<=70)
			x = map(x, 50, 70, 160, 126);
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
	
	public void setBiomeColor(GroundType gt, float soilMoisture) {
		label.setText(""+((int)(soilMoisture*100))+"%");
		rect.setFill(
				switch(gt) {
					case DIRT->soilMoisture<.5?Color.ROSYBROWN : Color.PERU; 
					case GRASS -> soilMoisture < .5 ? Color.LIGHTGREEN : Color.CHARTREUSE;
					case FOREST -> soilMoisture < .5 ? Color.FORESTGREEN : Color.GREEN;
					case ICE -> soilMoisture < .5 ? Color.ALICEBLUE : Color.PALETURQUOISE;
					case OCEAN -> Color.ROYALBLUE;
					case SAND -> soilMoisture < .5? Color.BLANCHEDALMOND : Color.KHAKI;
					case STONE -> soilMoisture < .5? Color.LIGHTGRAY : Color.SLATEGRAY;
					case LAKE -> Color.AQUA;
				} 
				);
	}

	public void setColorVisible(boolean isColor) {
		rect.setVisible(isColor);
	}

	public void setLabelVisible(boolean isNumerical) {
		label.setVisible(isNumerical);
	}

	//yes I named it that
	public void setSnowverlay(float amount) {
		label.setText(""+((int)(amount*100))+"%");
		rect.setFill(Color.ALICEBLUE.deriveColor(0, 1, 1, amount));
		label.setVisible(false);
		rect.setVisible(true);
	}

	public void setWind(float[] w) {
		float speed = MathUtils.dist(0f, 0f, 0f, w[0], w[1], w[2]);
		float mag = (float) Math.log(speed);
		float angle = (float) (Math.toDegrees(Math.atan2(w[1],w[0])) + 360) % 360;
		
		String dir = "";
		if(mag > .1) {
			dir += inRange(angle, 25 , 155) ? "N" : "";
			dir += inRange(angle, 270-65, 270+65) ? "S" : "";
			dir += inRange(angle, 0, 25) || inRange(angle, 360-65, 360) ? "E" : "";
			dir += inRange(angle, 180-65, 180+65) ? "W" : "";
		
			label.setText(String.format("%2s %1d", dir, (int)mag));
		}
		else
			label.setText("");
	}

	public void setColorEffect(GaussianBlur gaussianBlur) {
		rect.setEffect(gaussianBlur);
	}
	
	
	
}
