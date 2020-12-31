package app.spring;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.time.Duration;
import java.util.function.Supplier;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import app.GlobeData;
import app.HeadlessSimulation;

@SpringBootApplication
public class SpringAPI {
	private static HeadlessSimulation headlessSimulation;
	
	public static final File SAVE_LOCATION = new File(System.getProperty("user.dir") + "/world.sim");

	//TODO adjust
	private static final int DEF_LATITUDE_DIV = 256;
	private static final int DEF_LONGITUDE_DIV = DEF_LATITUDE_DIV * 2;
	private static final int DEF_ALTITUDE_DIV = 50;
	
	public static void launchSpring(String[] args) {
		headlessSimulation = new HeadlessSimulation(load(SAVE_LOCATION, ()->{ 
			return new GlobeData(DEF_LATITUDE_DIV, DEF_LONGITUDE_DIV, DEF_ALTITUDE_DIV); 
		}), 60_000); //by the minute updates
		headlessSimulation.autoSave(SAVE_LOCATION, Duration.ofHours(1));
		SpringApplication.run(SpringAPI.class, args);
	}

	private static GlobeData load(File f, Supplier<GlobeData> computedDefault) {
		if(!f.exists()) {
			return computedDefault.get();
		}
		try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f))){
			return (GlobeData) ois.readObject();
		}catch (Exception e) {
			e.printStackTrace();
			return computedDefault.get();
		}
	}
	
	public static HeadlessSimulation headlessSimulation() {
		return headlessSimulation;
	}
}
