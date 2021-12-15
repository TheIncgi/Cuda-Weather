package app.view;

import app.CudaUtils;
import app.util.Simulator;
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
			 renderClouds = new CheckBox("Clouds");
	Slider zoom = new Slider(.5, 10, 1);
	Slider elevation = new Slider(0, 1, 0);
	ToolBar top = new ToolBar(rerender, overlayPicker, renderClouds, renderSnow, zoom, elevation);
	/**Uses pointers from simulator to avoid copying data*/
	public GpuGlobeRenderView() {
		setTop(top);
		
		overlayPicker.getItems().addAll(Overlay.values());
		overlayPicker.getSelectionModel().select(Overlay.NONE);
		
		//TODO overlay controls
		
		rerender.setOnAction( e->update() );
		img = new WritableImage(IMAGE_WIDTH, IMAGE_HEIGHT);
		imageView.setPreserveRatio(true);
		
		setCenter(new ScrollPane(imageView));
		
		imageView.setImage(img);
		CudaUtils.init();
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
