package app;

import app.util.Simulator;
import app.view.GlobeViewer;
import javafx.application.Application;
import javafx.application.Platform;
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
	Simulator simulator;
	Thread bgThread;
	GlobeData result;
	
	volatile boolean running = true;
	private boolean flag = false;
	private volatile int timesteps = 0;
	@Override
	public void start(Stage stage) throws Exception {
		stage.setScene(new Scene(gv, 900, 600));
		gv.setBottom(simulationControls);
		
		
		timeStep.setOnAction(e->{
			flag = true;
			timeStep.setDisable(true);
			synchronized (bgThread) {
				bgThread.notify();
			}
		});
		
		bgThread = new Thread(()->{
			
			
			System.out.println("BG Thread started");
			
			System.out.println("Creating Cuda Context");
			CudaUtils.init();
			System.out.println("Creating simulation space");
			simulator = new Simulator(dummy);
			simulator.setOnStepComplete(()->{
				Platform.runLater(()->{
					timeStep.setDisable(false);
				});
			});
			simulator.setOnResultReady(()->{
				Platform.runLater(()->{
					gv.updateOverlays();
				});
			});
			
			while(running) {
				try {
					synchronized (bgThread) {
						bgThread.wait();
					}
					if(flag) {
						flag = false;
						long time = simulator.timeStep();
						System.out.printf("Timestep %d took %d millis\n", ++timesteps, time);
					}
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}
			}
			System.out.println("BG Thread has exited");
		});
		bgThread.start();
		
		stage.setOnCloseRequest(e->{
			running = false;
			flag = false;
			synchronized (bgThread) {
				bgThread.notify();
			}
		});
		
		stage.show();
		
	}
	
	
}
