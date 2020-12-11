package app;

import app.util.Simulator;
import app.view.GlobeViewer;
import app.view.LoadingPane;
import app.view.WorldSizePicker;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ToolBar;
import javafx.scene.image.Image;
import javafx.stage.Stage;

public class FxViewer extends Application{
	public static void main(String[] args) {
		launch();
	}
	
	//GlobeData dummy = new GlobeData(1500, 2500, 50).random();
	//GlobeViewer gv = new GlobeViewer(dummy);
	Button timeStep = new Button("Step");
	ProgressBar timestepProgress = new ProgressBar();
	Label timeStepStatus = new Label("Not ready");
	ToolBar simulationControls = new ToolBar(timeStep, timestepProgress, timeStepStatus);
	Simulator simulator;
	Thread bgThread;
	GlobeData result;
	
	WorldSizePicker wsp = new WorldSizePicker();
	
	volatile boolean running = true;
	private boolean flag = false;
	private volatile int timesteps = 0;
	private Scene scene;
	@Override
	public void start(Stage stage) throws Exception {
		stage.setScene(scene = new Scene(wsp, 900, 600));
		//gv.setBottom(simulationControls);
		
		wsp.setOnSelect(selectedWorldSize->{
			final LoadingPane lp = new LoadingPane();
			Thread buildWorld = new Thread(()->{
				result = new GlobeData(selectedWorldSize.a, selectedWorldSize.b, selectedWorldSize.c);
				TerrainGenerator tg = new TerrainGenerator();
				tg.setStepListener((step, progress)->Platform.runLater(()->lp.status(step, progress)));
				Platform.runLater(()->{
					lp.status("Fetching map...", 1);
				});
				
				tg.generate(result);
				CudaUtils.destroyContext();
				Platform.runLater(()->{
					lp.status("Building display...", 1);
					GlobeViewer gv = new GlobeViewer(result);
					launchSimThread(result, gv);
					scene.setRoot(gv);
					gv.setBottom(simulationControls);
				});
				
			}, "World Construction");
			buildWorld.start();
			scene.setRoot(lp);
		});
		
		timeStep.setOnAction(e->{
			System.out.println("Triggering timestep");
			flag = true;
			timeStep.setDisable(true);
			synchronized (bgThread) {
				bgThread.notify();
			}
		});
		timeStep.setDisable(true);
		//timestepProgress.setVisible(false);
		
		stage.setOnCloseRequest(e->{
			running = false;
			flag = false;
			if(bgThread!=null)
			synchronized (bgThread) {
				bgThread.notify();
			}
		});
		stage.setTitle("Weather with CUDA - TheIncgi");
		stage.getIcons().add(new Image(FxViewer.class.getResourceAsStream("storm32.png")));
		stage.show();
		
	}
	
	private void launchSimThread(final GlobeData world, final GlobeViewer gv) {
		bgThread = new Thread(()->{
			
			
			System.out.println("BG Thread started");
			
			System.out.println("Creating Cuda Context");
			CudaUtils.init();
			System.out.println("Creating simulation space");
			simulator = new Simulator(world);
			System.out.println("  adding simulation listeners");
			simulator.setProgressListener((prog, status)->{
				Platform.runLater(()->{
					timestepProgress.setProgress(prog);
					timeStepStatus.setText(status);
					//timestepProgress.setVisible(prog<1);
				});
			});
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
			
			System.out.println("BG - Waiting for trigger");
			Platform.runLater(()->{
				timeStep.setDisable(false);
				timeStepStatus.setText("Ready");
			});
			while(running) {
				try {
					synchronized (bgThread) {
						bgThread.wait();
						System.out.println("BG - Notified");
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
			System.out.println("Closing resources...");
			simulator.close();
			System.out.println("BG Thread has exited");
		}, "Simulation Thread");
		bgThread.start();
	}
	
}
