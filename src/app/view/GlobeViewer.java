package app.view;

import app.GlobeData;
import app.util.MathUtils;
import javafx.beans.value.ObservableValue;
import javafx.scene.Node;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Slider;
import javafx.scene.control.ToolBar;
import javafx.scene.effect.GaussianBlur;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;

public class GlobeViewer extends BorderPane{
	Pane layers = new Pane();
	ScrollPane scrolly = new ScrollPane(layers);


	private static final float OVERLAY_OPACITY = .6f;

	HashedGridPane terainGrid = new HashedGridPane("terrain"), 
			tempGrid = new HashedGridPane("temp"), 
			humidityGrid = new HashedGridPane("humid"), 
			cloudCoverGridAlti = new HashedGridPane("cloudAlt"), 
			cloudCoverGridAll = new HashedGridPane("cloudAll"), 
			windGrid = new HashedGridPane("wind"), 
			snowGrid = new HashedGridPane("snow"), 
			rainGrid = new HashedGridPane("rain"); //this one is calculated from humidity

	ComboBox<String> colorOverlay = new ComboBox<>(), numberOverlay = new ComboBox<>();
	CheckBox showSnow = new CheckBox("Show Snow");
	Slider altitudeSlider;
	ToolBar layerModes = new ToolBar();
	private GlobeData globeData;
	Label debug = new Label("debug");

	public GlobeViewer(GlobeData gd) {
		this.globeData = gd;
		for(int la=0; la<gd.latitudeDivisions; la++)
			for(int lo=0; lo<gd.longitudeDivisions; lo++) {
				FloatVisulaizationTile terain = new FloatVisulaizationTile();
				FloatVisulaizationTile temp = new FloatVisulaizationTile();
				FloatVisulaizationTile humid = new FloatVisulaizationTile();
				FloatVisulaizationTile cloud = new FloatVisulaizationTile();
				FloatVisulaizationTile cloud2 = new FloatVisulaizationTile();
				FloatVisulaizationTile wind = new FloatVisulaizationTile();
				FloatVisulaizationTile snow = new FloatVisulaizationTile();
				FloatVisulaizationTile rain = new FloatVisulaizationTile();

				temp.setOpacity(OVERLAY_OPACITY);
				humid.setOpacity(OVERLAY_OPACITY);
				cloud.setOpacity(OVERLAY_OPACITY);
				wind.setOpacity(OVERLAY_OPACITY);
				rain.setOpacity(OVERLAY_OPACITY);
				snow.setOpacity(OVERLAY_OPACITY + (1-OVERLAY_OPACITY)/2);
				snow.setColorEffect(new GaussianBlur(8));
				cloud.setColorEffect(new GaussianBlur(4));
				cloud2.setColorEffect(new GaussianBlur(4));

				terain.setBiomeColor(gd.groundType[la][lo], gd.groundMoisture[la][lo]);



				terainGrid.add(terain, lo, la);
				tempGrid.add(temp, lo, la);
				humidityGrid.add(humid, lo, la);
				cloudCoverGridAll.add(cloud, lo, la);
				cloudCoverGridAlti.add(cloud2, lo, la);
				windGrid.add(wind, lo, la);
				snowGrid.add(snow, lo, la);
				rainGrid.add(rain, lo, la);

				terain.setColorVisible(true);
				terain.setLabelVisible(false);
				snow.setLabelVisible(false);
				snow.setColorVisible(true);
			}

		layers.getChildren().addAll(terainGrid, snowGrid);

		

		numberOverlay.getItems().addAll("None","Humidity", "Temperature", "Wind Speed", "Cloud Coverage-Alti","Cloud Coverage-All", "Percipitation");
		colorOverlay.getItems().addAll("None","Humidity", "Temperature", "Wind Speed", "Cloud Coverage-Alti","Cloud Coverage-All", "Percipitation");
		numberOverlay.getSelectionModel().select(0);
		colorOverlay.getSelectionModel().select(0);
		altitudeSlider = new Slider(0, gd.altitudeDivisions-.5, 0);
		numberOverlay.valueProperty().addListener(e->onChangedMode());
		colorOverlay.valueProperty().addListener(e->onChangedMode());
		altitudeSlider.valueProperty().addListener(e->onChangedMode());

		layerModes.getItems().addAll(showSnow, altitudeSlider, new Label("Color-Number"),colorOverlay, numberOverlay);
		showSnow.setSelected(true);
		showSnow.setOnAction(e->{
			if(showSnow.isSelected() && !layers.getChildren().contains(snowGrid))
				layers.getChildren().add(1, snowGrid); 
			else 
				layers.getChildren().remove(snowGrid);});

		this.setTop(layerModes);
		this.setCenter(scrolly);
		this.setRight(debug);

		updateOverlays();
	}
	
	public void replaceGlobeData(GlobeData gd) {
		this.globeData = gd;
		updateOverlays();
	}

	public void updateOverlays() {
		int altitude = (int) altitudeSlider.getValue();
		float mixup = (float) (altitudeSlider.getValue() % 1);
		for(int lat = 0; lat < globeData.latitudeDivisions; lat++) {
			for(int lon = 0; lon < globeData.longitudeDivisions; lon++) {
				if(cloudCoverGridAll.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) {
					if(cloudCoverGridAll.isVisible()) {
						float cu = 0;
						for (int alti = 0; alti < globeData.altitudeDivisions; alti++) 
							cu = Math.max(globeData.cloudCover[lat][lon][alti], cu);


						fvt.setPercentColor(cu);
					}
				}
				if(humidityGrid.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) {
					if(humidityGrid.isVisible()) {
						fvt.setPercentColor(globeData.humidity[lat][lon][altitude]);
					}
				}
				if(rainGrid.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) 
					if(rainGrid.isVisible()) {
						fvt.setPercentColor(globeData.percipitationChanceAt(lat, lon));
					}
				if(snowGrid.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) 
					if(snowGrid.isVisible()) {
						fvt.setSnowverlay(globeData.getSnowCoveragePercent(lat, lon));
					}
				if(tempGrid.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) 
					if(tempGrid.isVisible()) {
						fvt.setTempColor(globeData.temp[lat][lon][altitude]);
					}
				if(windGrid.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) 
					if(windGrid.isVisible()) {
						fvt.setWind(globeData.windSpeed[lat][lon][altitude]);
					}

			}
		}
	}

	public void onChangedMode() {
		layers.getChildren().clear();
		layers.getChildren().addAll(terainGrid, snowGrid);
		HashedGridPane col = 
				switch(colorOverlay.getValue()) {
				case "Humidity" -> humidityGrid;
				case "Temperature" -> tempGrid;
				case "Wind Speed" -> windGrid;
				case "Cloud Coverage-Alti" -> cloudCoverGridAlti;
				case "Cloud Coverage-All"  -> cloudCoverGridAll;
				case "Percipitation" -> rainGrid;
				default -> null;
				};
		HashedGridPane num = 
				switch(numberOverlay.getValue()) {
				case "Humidity" -> humidityGrid;
				case "Temperature" -> tempGrid;
				case "Wind Speed" -> windGrid;
				case "Cloud Coverage-Alti" -> cloudCoverGridAlti;
				case "Cloud Coverage-All"  -> cloudCoverGridAll;
				case "Percipitation" -> rainGrid;
				default -> null;
		};

		if(col!=null)
			col.getChildren().forEach(n->{
				if(n instanceof FloatVisulaizationTile fvt) {
					fvt.setColorVisible(true);
					fvt.setLabelVisible(num==col);
				}
			});
		if(col!=num && num!=null) {
			num.getChildren().forEach(n->{
				if(n instanceof FloatVisulaizationTile fvt) {
					fvt.setColorVisible(false);
					fvt.setLabelVisible(true);
				}
			});
		}
		if(col!=null)
			layers.getChildren().add(col);
		if(num!=null && col!=num)
			layers.getChildren().add(num);

		updateOverlays();
	}
}
