package com.bytecraft;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.web.servlet.support.SpringBootServletInitializer;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ByteCraftApplication extends SpringBootServletInitializer {

    @Override
    protected SpringApplicationBuilder configure(SpringApplicationBuilder application) {
        return application.sources(ByteCraftApplication.class);
    }

	public static void main(String[] args) {
		SpringApplication.run(ByteCraftApplication.class, args);
	}

}
