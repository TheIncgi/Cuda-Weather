package app.view;

import app.GlobeData;
import app.TerrainGenerator;
import app.util.MathUtils;
import javafx.application.Platform;
import javafx.beans.value.ObservableValue;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Slider;
import javafx.scene.control.ToolBar;
import javafx.scene.effect.GaussianBlur;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;

public class GlobeViewer extends BorderPane{
	Pane layers = new Pane();
	ScrollPane scrolly = new ScrollPane(layers);


	private static final float OVERLAY_OPACITY = .6f;

	HashedGridPane terainGrid = new HashedGridPane("terrain"), 
			elevGrid = new HashedGridPane("elevation"), 
			tempGrid = new HashedGridPane("temp"), 
			humidityGrid = new HashedGridPane("humid"), 
			cloudCoverGridAlti = new HashedGridPane("cloudAlt"), 
			cloudCoverGridAll = new HashedGridPane("cloudAll"), 
			windGrid = new HashedGridPane("wind"), 
			snowGrid = new HashedGridPane("snow"), 
			pressure = new HashedGridPane("pressure"),
			rainGrid = new HashedGridPane("rain"); //this one is calculated from humidity

	ComboBox<String> colorOverlay = new ComboBox<>(), numberOverlay = new ComboBox<>();
	CheckBox showSnow = new CheckBox("Show Snow");
	Slider altitudeSlider, zoomSlider;
	ToolBar layerModes = new ToolBar();
	private GlobeData globeData;
	Label debug = new Label("debug");
	DebugPanel debugTile = new DebugPanel("Pos:","Latitude:","Longitude:","Altitude:","Elevation:", "Ground:", "Snow:", "Temp:","Humidity:","Cloud:", "Cloud Cover:", "Wind:", "Rain:", "Pressure:");
	Button regenerate = new Button("Regen Terrain");
	Button testButton = new Button("Test");
	VBox rightPanel = new VBox(debug, debugTile, regenerate, testButton);
	
	public GlobeViewer(GlobeData gd) {
		this.globeData = gd;
		for(int la=0; la<gd.latitudeDivisions; la++)
			for(int lo=0; lo<gd.longitudeDivisions; lo++) {
				FloatVisulaizationTile terain = new FloatVisulaizationTile(lo, la);
				FloatVisulaizationTile elev = new FloatVisulaizationTile(lo, la);
				FloatVisulaizationTile temp = new FloatVisulaizationTile(lo, la);
				FloatVisulaizationTile humid = new FloatVisulaizationTile(lo, la);
				FloatVisulaizationTile cloud = new FloatVisulaizationTile(lo, la);
				FloatVisulaizationTile cloud2 = new FloatVisulaizationTile(lo, la);
				FloatVisulaizationTile wind = new FloatVisulaizationTile(lo, la);
				FloatVisulaizationTile snow = new FloatVisulaizationTile(lo, la);
				FloatVisulaizationTile rain = new FloatVisulaizationTile(lo, la);
				FloatVisulaizationTile pres = new FloatVisulaizationTile(lo, la);

				temp.setOpacity(OVERLAY_OPACITY);
				elev.setOpacity(OVERLAY_OPACITY + (1-OVERLAY_OPACITY)/2);
				humid.setOpacity(OVERLAY_OPACITY);
				cloud.setOpacity(OVERLAY_OPACITY);
				cloud2.setOpacity(OVERLAY_OPACITY);
				wind.setOpacity(OVERLAY_OPACITY);
				rain.setOpacity(OVERLAY_OPACITY);
				snow.setOpacity(OVERLAY_OPACITY + (1-OVERLAY_OPACITY)/2);
				pres.setOpacity(OVERLAY_OPACITY);
				
				snow.setColorEffect(new GaussianBlur(8));
				cloud.setColorEffect(new GaussianBlur(4));
				cloud2.setColorEffect(new GaussianBlur(4));

				terain.setBiomeColor(GlobeData.GroundType.values()[gd.groundType[la][lo]], gd.groundMoisture[la][lo]);

				terain.setOnMouseEntered(e->onMouseOver(terain));
				elev.setOnMouseEntered(e->onMouseOver(elev));
				temp.setOnMouseEntered(e->onMouseOver(terain));
				humid.setOnMouseEntered(e->onMouseOver(terain));
				cloud.setOnMouseEntered(e->onMouseOver(terain));
				cloud2.setOnMouseEntered(e->onMouseOver(terain));
				wind.setOnMouseEntered(e->onMouseOver(terain));
				snow.setOnMouseEntered(e->onMouseOver(terain));
				pres.setOnMouseEntered(e->onMouseOver(terain));
				rain.setOnMouseEntered(e->onMouseOver(terain));

				terainGrid.add(terain, lo, la);
				elevGrid.add(elev, lo, la);
				tempGrid.add(temp, lo, la);
				humidityGrid.add(humid, lo, la);
				cloudCoverGridAll.add(cloud, lo, la);
				cloudCoverGridAlti.add(cloud2, lo, la);
				windGrid.add(wind, lo, la);
				snowGrid.add(snow, lo, la);
				pressure.add(pres, lo, la);
				rainGrid.add(rain, lo, la);

				terain.setColorVisible(true);
				terain.setLabelVisible(false);
				snow.setLabelVisible(false);
				snow.setColorVisible(true);
			}

		layers.getChildren().addAll(terainGrid);

		

		numberOverlay.getItems().addAll("None","Humidity", "Temperature", "Wind Speed", "Cloud Coverage-Alti","Cloud Coverage-All", "Pressure", "Percipitation");
		colorOverlay.getItems().addAll("None","Elevation","Humidity", "Temperature", "Wind Speed", "Cloud Coverage-Alti","Cloud Coverage-All", "Pressure", "Percipitation");
		numberOverlay.getSelectionModel().select(0);
		colorOverlay.getSelectionModel().select(0);
		altitudeSlider = new Slider(0, gd.altitudeDivisions-.5, 0);
		numberOverlay.valueProperty().addListener(e->onChangedMode());
		colorOverlay.valueProperty().addListener(e->onChangedMode());
		altitudeSlider.valueProperty().addListener(e->onChangedMode());
		zoomSlider = new Slider(1, 20, 1);
		zoomSlider.valueProperty().addListener(v->{
			layers.setScaleX(1/zoomSlider.getValue());
			layers.setScaleY(1/zoomSlider.getValue());
			layers.setTranslateX( -( layers.getWidth() / 2 - layers.getWidth() * layers.getScaleX() / 2 ) );
			layers.setTranslateY( -( layers.getHeight() / 2 - layers.getHeight() * layers.getScaleY() / 2 ) );
		});
		
		regenerate.setOnAction(e->{
			regenerate.setDisable(true);
			Thread buildWorld = new Thread(()->{
				System.out.println("Regenerating...");
				GlobeData result = new GlobeData(globeData.latitudeDivisions, globeData.longitudeDivisions, globeData.altitudeDivisions);
				TerrainGenerator tg = new TerrainGenerator();
				tg.generate(result);
				System.out.println("Loading regenerated terrain..");
				Platform.runLater(()->{
					GlobeViewer.this.replaceGlobeData(result);
					regenerate.setDisable(false);
				});
				
			}, "World Construction");
			buildWorld.start();
		});
		
		testButton.setOnAction(e->{
			float sum = 0;
			for (int la = 0; la < globeData.latitudeDivisions; la++) {
				for (int lo = 0; lo < globeData.longitudeDivisions; lo++) {
					for (int al = 0; al < globeData.altitudeDivisions; al++) {
						sum += globeData.windSpeed[la][lo][al*3]; //debug, holds air mass rn
					}
				}
			}
			testButton.setText(sum+"");
		});
		
		layerModes.getItems().addAll(showSnow, new Label("Altitude: "),altitudeSlider, new Label("Zoom:"), zoomSlider, new Label("Color-Number"),colorOverlay, numberOverlay);
		showSnow.setSelected(false);
		showSnow.setOnAction(e->{
			if(showSnow.isSelected() && !layers.getChildren().contains(snowGrid))
				layers.getChildren().add(1, snowGrid); 
			else 
				layers.getChildren().remove(snowGrid);});

		this.setTop(layerModes);
		this.setCenter(scrolly);
		this.setRight(rightPanel);

		updateOverlays();
	}
	
	public void replaceGlobeData(GlobeData gd) {
		System.out.println("Updating map...");
		this.globeData = gd;
		for(int lat = 0; lat < globeData.latitudeDivisions; lat++) {
			for(int lon = 0; lon < globeData.longitudeDivisions; lon++) {
				if(terainGrid.getNode(lon, lat) instanceof FloatVisulaizationTile fvt)
					fvt.setBiomeColor(GlobeData.GroundType.values()[gd.groundType[lat][lon]], gd.groundMoisture[lat][lon]);
			}
		}
		
		
		updateOverlays();
		System.out.println("Replacement complete");
	}
	
	
	public void updateOverlays() {
		debug.setText(String.format("World time: \n%8.4f\n%8.4f\n%s", globeData.time[0], globeData.time[1], globeData.getTime(0)));
		
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
				if(cloudCoverGridAlti.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) {
					if(fvt.isVisible())
						fvt.setPercentColor(globeData.cloudCover[lat][lon][altitude]);
				}
				if(humidityGrid.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) {
					if(humidityGrid.isVisible()) {
						fvt.setPercentColor(globeData.humidity[lat][lon][altitude]);
					}
				}
				if(elevGrid.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) {
					if(elevGrid.isVisible())
						fvt.setTempColor(MathUtils.map(globeData.elevation[lat][lon], -.5f, 6f, -20f, 100f));
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
						fvt.setWind(globeData.windSpeed[lat][lon][altitude*3],globeData.windSpeed[lat][lon][altitude*3+1],globeData.windSpeed[lat][lon][altitude*3+2]);
					}
				if(pressure.getNode(lon, lat) instanceof FloatVisulaizationTile fvt) 
					if(pressure.isVisible()) {
						fvt.setPercentColor(globeData.pressure[lat][lon][altitude]/1.2f);
					}

			}
		}
	}

	public void onChangedMode() {
		layers.getChildren().clear();
		layers.getChildren().add(terainGrid);
		if(showSnow.isSelected())
			layers.getChildren().add(snowGrid);
		HashedGridPane col = 
				switch(colorOverlay.getValue()) {
				case "Humidity" -> humidityGrid;
				case "Temperature" -> tempGrid;
				case "Wind Speed" -> windGrid;
				case "Cloud Coverage-Alti" -> cloudCoverGridAlti;
				case "Cloud Coverage-All"  -> cloudCoverGridAll;
				case "Percipitation" -> rainGrid;
				case "Pressure" -> pressure;
				case "Elevation" -> elevGrid;
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
				case "Pressure" -> pressure;
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
	
	private void onMouseOver(FloatVisulaizationTile fvt) {
		int alti = (int) altitudeSlider.getValue();
		float cu = 0;
		for (int a = 0; a < globeData.altitudeDivisions; a++) 
			cu = Math.max(globeData.cloudCover[fvt.y][fvt.x][a], cu);
		
		debugTile.changeValue("Pos:", 			String.format("%d %d", fvt.x, fvt.y));
		debugTile.changeValue("Latitude:", 		String.format("%7.4f°", fvt.y * -180d / globeData.latitudeDivisions +90));
		debugTile.changeValue("Longitude:", 	String.format("%7.4f°", fvt.x * 360d / globeData.longitudeDivisions));
		debugTile.changeValue("Elevation:", 	String.format("%5.1fm", globeData.elevation[fvt.y][fvt.x]));
		debugTile.changeValue("Altitude:", 		String.format("%7.4f",  globeData.altitudeLowerBound(alti)));
		debugTile.changeValue("Ground:", 		String.format("%s", GlobeData.GroundType.values()[globeData.groundType[fvt.y][fvt.x]]));
		debugTile.changeValue("Snow:", 			String.format("%4.1fcm", globeData.snowCover[fvt.y][fvt.x]));
		debugTile.changeValue("Temp:", 			String.format("%3.0f°F", globeData.temp[fvt.y][fvt.x][alti]));
		debugTile.changeValue("Humidity:", 		String.format("%3.0f%%",   globeData.humidity[fvt.y][fvt.x][alti] * 100));
		debugTile.changeValue("Cloud:", 		String.format("%3.0f%%",   globeData.cloudCover[fvt.y][fvt.x][alti] * 100));
		debugTile.changeValue("Cloud Cover:", 	String.format("%3.0f%%",   cu * 100)); //not accounting for angle of sun
		debugTile.changeValue("Wind:", 			String.format("%5.2f\n%5.2f\n%5.2f",   globeData.windSpeed[fvt.y][fvt.x][alti*3],globeData.windSpeed[fvt.y][fvt.x][alti*3+1],globeData.windSpeed[fvt.y][fvt.x][alti*3+2]));
		debugTile.changeValue("Rain:", 			String.format("%3.0f%%S", globeData.percipitationChanceAt(fvt.y, fvt.x) * 100));
		debugTile.changeValue("Pressure:", 		String.format("%4.3f",   globeData.pressure[fvt.y][fvt.x][alti]));
	}
}
