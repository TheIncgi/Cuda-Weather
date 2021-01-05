package app.spring;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.time.Duration;
import java.util.function.Supplier;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import app.CudaUtils;
import app.GlobeData;
import app.HeadlessSimulation;
import app.TerrainGenerator;

@SpringBootApplication
public class SpringAPI {
	private static HeadlessSimulation headlessSimulation;
	
	public static final File SAVE_LOCATION = new File(System.getProperty("user.dir") + "/world.sim");

	//TODO adjust
	private static final int DEF_LATITUDE_DIV = 144;
	private static final int DEF_LONGITUDE_DIV = DEF_LATITUDE_DIV * 2;
	private static final int DEF_ALTITUDE_DIV = 70;
	
	public static void main(String[] args) {
		launchSpring(args);
	}
	
	public static void launchSpring(String[] args) {
		headlessSimulation = new HeadlessSimulation(load(SAVE_LOCATION, ()->{ 
			GlobeData gd = new GlobeData(DEF_LATITUDE_DIV, DEF_LONGITUDE_DIV, DEF_ALTITUDE_DIV);
			CudaUtils.init();
			TerrainGenerator tg = new TerrainGenerator();
			tg.generate(gd);
			CudaUtils.destroyContext();
			return gd;
		}), 60_000); //by the minute updates
		headlessSimulation.autoSave(SAVE_LOCATION, Duration.ofHours(1));
		System.out.println("Launching Spring server");
		SpringApplication.run(SpringAPI.class, args);
	}

	private static GlobeData load(File f, Supplier<GlobeData> computedDefault) {
		if(!f.exists()) {
			System.out.println("Creating a new world...");
			return computedDefault.get();
		}
		try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f))){
			System.out.println("Loading world...");
			return (GlobeData) ois.readObject();
		}catch (Exception e) {
			e.printStackTrace();
			System.out.println("Creating a new world...");
			return computedDefault.get();
		}
	}
	
	public static HeadlessSimulation headlessSimulation() {
		return headlessSimulation;
	}
}
