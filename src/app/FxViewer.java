package app;

import app.util.Simulator;
import app.view.GlobeViewer;
import app.view.LoadingPane;
import app.view.WorldSizePicker;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
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
//	Button blindStep = new Button("Blind Step");
	ComboBox<Integer> stepCount = new ComboBox<>();
	Button timeStep = new Button("Step");
	Button cancelButton = new Button("Cancel");
	ProgressBar timestepProgress = new ProgressBar();
	Label timeStepStatus = new Label("Not ready");
	ToolBar simulationControls = new ToolBar(new Label("Steps: "),stepCount, timeStep, timestepProgress, timeStepStatus, cancelButton);
	Simulator simulator;
	Thread bgThread;
	GlobeData result;
	
	WorldSizePicker wsp = new WorldSizePicker();
	
	volatile boolean running = true, cancel = false;
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
		
		stepCount.getItems().addAll(1, 5, 10, 15, 20, 25, 30, 50, 100, 125, 150, 300, 500, 1000, 2000, 5000, 10000, 20000,25000,50000,75000,100000);
		stepCount.getSelectionModel().select(0);
		
		timeStep.setOnAction(e->{
			System.out.println("Triggering timestep");
			flag = true;
			timeStep.setDisable(true);
			synchronized (bgThread) {
				bgThread.notify();
			}
		});
		cancelButton.setOnAction(e->{
			cancel = true;
			cancelButton.setDisable(true);
		});
		cancelButton.setVisible(false);
		
		timeStep.setDisable(true);
		//timestepProgress.setVisible(false);
		
		stage.setOnCloseRequest(e->{
			running = false;
			cancel = true;
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
				if(stepCount.getValue()==1)
					Platform.runLater(()->{
						timestepProgress.setProgress(prog);
						timeStepStatus.setText(status);
						//timestepProgress.setVisible(prog<1);
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
						cancel = false;
						Platform.runLater(()->{
							cancelButton.setVisible(true);
							cancelButton.setDisable(false);
						});
						
						long time = 0;
						int limit = stepCount.getValue();
						for(int i = 0; i < limit && running && !cancel; i++) {
							if(limit != 1) {
								final int I = i;
								Platform.runLater(()->{
									timestepProgress.setProgress(I / (float)limit);
									timeStepStatus.setText(String.format("Step %5d of %d", I+1, limit));
								});
							}
							time += simulator.timeStep( i == limit-1);
						}
						System.out.printf("Timestep"+(limit!=1? (stepCount.getValue()==1?"":"s"): "")+" %d took %d millis\n", ++timesteps, time);
//						if(limit!=1)
							Platform.runLater(()->{
								timeStep.setDisable(false);
								timeStepStatus.setText("Ready");
								timestepProgress.setProgress(-1);
							});
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
