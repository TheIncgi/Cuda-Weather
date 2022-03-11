package app.view;

import app.CudaUtils;
import app.util.Simulator;
import javafx.event.ActionEvent;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Slider;
import javafx.scene.control.ToolBar;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.Pane;

public class GpuGlobeRenderView extends BorderPane{
	private Simulator simulator;
	
	private ImageView imageView = new ImageView();
	public static final int IMAGE_WIDTH = 1920;
	public static final int IMAGE_HEIGHT = 1080;
	private final int[] buffer = new int[IMAGE_WIDTH*IMAGE_HEIGHT];
	private WritableImage img;
	Button rerender = new Button("Re-render");
	ComboBox<Overlay> overlayPicker = new ComboBox<>();
	CheckBox renderSnow = new CheckBox("Snow"),
			 renderClouds = new CheckBox("Clouds"),
			 renderSunshine = new CheckBox("Sunshine");
	Slider zoom = new Slider(.5, 10, 1);
	Slider elevation = new Slider(0, 1, 0);
	Button resetSim = new Button("Reset Sim");
	ToolBar top = new ToolBar(rerender, overlayPicker, renderClouds, renderSnow, renderSunshine, zoom, elevation, resetSim);
	/**Uses pointers from simulator to avoid copying data*/
	public GpuGlobeRenderView() {
		setTop(top);
		
		overlayPicker.getItems().addAll(Overlay.values());
		overlayPicker.getSelectionModel().select(Overlay.NONE);
		
		addControlEvents();
		
		rerender.setOnAction( e->update() );
		img = new WritableImage(IMAGE_WIDTH, IMAGE_HEIGHT);
		imageView.setPreserveRatio(true);
		
		setCenter(new ScrollPane(imageView));
		
		imageView.setImage(img);
		imageView.setOnMouseMoved(e->{
			simulator.setMousePos((int) (e.getSceneX() - imageView.getLayoutX()), (int) (e.getSceneY()-imageView.getLayoutY()));
		});
		CudaUtils.init();
	}
	

	private void addControlEvents() {
		overlayPicker.setOnAction(this::updateFlags);
		renderSnow.setOnAction(this::updateFlags);
		renderClouds.setOnAction(this::updateFlags);
		renderSunshine.setOnAction(this::updateFlags);
		
		resetSim.setOnAction(e->{
			simulator.initAtmosphere(true);
			rerender.fire();
		});
	}
	
	private void updateFlags(ActionEvent e) {
		simulator.setOverlayFlags( getRenderFlags() );
	}

	public void setSimulator(Simulator simulator) {
		this.simulator = simulator;
	}
	
	public void update() {
		System.out.println("Rendering...");
		simulator.render(buffer);
		img.getPixelWriter().setPixels(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, 
				PixelFormat.getIntArgbInstance(), buffer, 0, IMAGE_WIDTH);	
	}
	
	public int getRenderFlags() {
		int flags = 0;
		flags |= renderSunshine.isSelected()? 1 : 0;
		flags |=   renderClouds.isSelected()? 2 : 0;
		flags |=     renderSnow.isSelected()? 4 : 0;
		
		var overlay = overlayPicker.getSelectionModel().getSelectedItem();
		if(!overlay.equals(Overlay.NONE))
			flags |= overlay.ordinal() << 3;
		
		return flags;
	}
	
	public enum Overlay {
		NONE,
		THERMAL,
		HUMIDITY,
		WIND,
		SNOW_COVERAGE,
		PERCIPITATION,
		ELEVATION
	}
}
