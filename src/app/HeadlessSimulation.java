package app;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.time.Duration;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Consumer;

import app.util.Simulator;
import jcuda.driver.JCudaDriver;


/**This class sets up a simulation that automatically updates itself at a fixed rate<br>
 * The simulation runs at 1x speed (real time) or as close as it can.<br>
 * The simulation will start as soon as an instance of this class is created.<br>
 * */
public class HeadlessSimulation {
	private GlobeData data;
	private Simulator simulator;
	private ReentrantReadWriteLock lock = new ReentrantReadWriteLock(true);
	private long simulationStart = System.currentTimeMillis();
	private long stepsCompleted = 0;
	/**Time in milliseconds between each batch of timesteps*/
	public final long updatePeriod;
	private Timer timer;
	private Timer saveTimer;
	
	/**Triggers before the write lock is released.*/
	private Runnable onTimestepBatchComplete;
	
	
	
	public HeadlessSimulation(int latitudeDivisions, int longitudeDivisions, int altitudeDivisions, long updatePeriod) {
		this( new GlobeData(latitudeDivisions, longitudeDivisions, altitudeDivisions), updatePeriod );
	}
	
	public HeadlessSimulation(GlobeData globeData, long updatePeriod) {
		this.updatePeriod = updatePeriod;
				
		timer = new Timer(true);
		saveTimer = new Timer(true);
		timer.scheduleAtFixedRate(new TimerTask() {
			@Override public void run() {
				timestep();
			}
		},  0, updatePeriod);
		
		lock.writeLock().lock();
		simulator.initAtmosphere();
		lock.writeLock().unlock();
	}
	
	private void timestep() {
		lock.writeLock().lock();
		
		long target = (long) ((System.currentTimeMillis() - simulationStart) / simulator.getTimestepSize()); //step count we should be at based on time ellapsed
		long maxBatchSize = (long) (updatePeriod / 1000 / simulator.getTimestepSize() * 2); //twice the number of steps in an update cycle
		int  limit  = (int) Math.min( target - stepsCompleted , maxBatchSize );
		if(limit==maxBatchSize) {
			System.err.printf("""
					[HeadlessSimulation] WARNING: Simulation is falling behind. World size may be to large or the scheduled task can not lock the Reentrant lock quickly enough.
					                              Simulation is behind %d steps.\n
					""", 
					target - (stepsCompleted + maxBatchSize/2)
			);
					
					
		
		}
		for(int i = 1; i <= limit; i++ )
			simulator.timeStep( i == limit );
		JCudaDriver.cuCtxSynchronize();
		stepsCompleted += limit;
		
		if(onTimestepBatchComplete!=null)
			onTimestepBatchComplete.run();
		
		lock.writeLock().unlock();
	}
	
	/**
	 * Allows access to the simulations GlobeData in a synchronized way.<br>
	 * ReenterantReadWriteLock is read locked during this operation.
	 * */
	public void getState(Consumer<GlobeData> consumer) {
		lock.readLock().lock();
		consumer.accept(data);
		lock.readLock().unlock();
	}
	
	/**
	 * Returns the GlobeData used in the simulation.<br>
	 * Use {@link HeadlessSimulation#readLock()} and {@link HeadlessSimulation#readUnlock()} to block
	 * changes during state reading.
	 * */
	public GlobeData getData() {
		return data;
	}
	
	public void readLock() {
		lock.readLock().lock();
	}
	public void readUnlock() {
		lock.readLock().unlock();
	}
	
	private boolean isSaveScheduled = false;
	/**
	 * Sets a timer task for saving.
	 * */
	public void autoSave(File file, Duration period) {
		if(isSaveScheduled) {
			saveTimer.cancel();
			saveTimer = new Timer(true);
		}
		saveTimer.scheduleAtFixedRate(new TimerTask() {
			@Override public void run() {
				try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))){
					lock.readLock().lock();
					oos.writeObject(data);
					lock.readLock().unlock();
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}, period.toMillis(), period.toMillis());
	}
	
	/**Set a task to complete before the write lock is released on a timestep batch*/
	public void setOnTimestepBatchComplete(Runnable onTimestepBatchComplete) {
		this.onTimestepBatchComplete = onTimestepBatchComplete;
	}
	
	/**Stops the simulation's scheduled task*/
	public void shutdown() {
		timer.cancel();
	}
}
