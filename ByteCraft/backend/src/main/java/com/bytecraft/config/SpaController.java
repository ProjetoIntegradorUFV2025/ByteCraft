package com.bytecraft.config;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

/** Only UI routes are forwarded; API and missing asset errors keep their status. */
@Controller
public class SpaController {
    @GetMapping({"/", "/aluno", "/professor", "/professor/cadastro", "/sala", "/niveis", "/fases", "/montagem", "/montagem-interna", "/importar", "/quiz"})
    public String index() {
        return "forward:/index.html";
    }
}
