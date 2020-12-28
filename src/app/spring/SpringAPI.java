package app.spring;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
public class SpringAPI {
	public static void launchSpring(String[] args) {
		SpringApplication.run(SpringAPI.class, args);
	}

}
