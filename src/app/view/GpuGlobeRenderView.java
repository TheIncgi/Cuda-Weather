package app.view;

import app.CudaUtils;
import app.util.Simulator;
import javafx.scene.control.Button;
import javafx.scene.control.ScrollPane;
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
	ToolBar top = new ToolBar(rerender);
	/**Uses pointers from simulator to avoid copying data*/
	public GpuGlobeRenderView() {
		setTop(top);
		rerender.setOnAction( e->update() );
		img = new WritableImage(IMAGE_WIDTH, IMAGE_HEIGHT);
		Pane wrapper = new Pane(imageView);
		imageView.setPreserveRatio(true);
		imageView.fitWidthProperty().bind(wrapper.widthProperty());
		wrapper.setPrefWidth(200);
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
}
