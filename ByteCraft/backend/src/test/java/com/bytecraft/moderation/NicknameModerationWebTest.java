package com.bytecraft.moderation;

import com.bytecraft.controller.AlunoController;
import com.bytecraft.security.SecurityConfig;
import com.bytecraft.service.AlunoService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.context.annotation.Import;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest({NicknameModerationController.class, AlunoController.class})
@Import({NicknameModerationService.class, SecurityConfig.class})
class NicknameModerationWebTest {
    @Autowired MockMvc mvc;
    @MockitoBean AlunoService alunos;

    @Test
    void exposesModerationWithSecurityEnabled() throws Exception {
        mvc.perform(post("/api/moderation/nickname").contentType("application/json")
            .content("{\"nickname\":\"puta\"}"))
            .andExpect(status().isOk()).andExpect(jsonPath("$.status").value("inadequado"));
        verifyNoInteractions(alunos);
    }
    @Test
    void directLoginCannotBypassModerationOrWriteToDatabase() throws Exception {
        mvc.perform(post("/api/alunos/login").contentType("application/json")
            .content("{\"apelido\":\"puta\",\"codigoSala\":\"1\"}"))
            .andExpect(status().isBadRequest()).andExpect(jsonPath("$.moderation").value("inadequado"));
        verifyNoInteractions(alunos);
    }
    @Test
    void invalidModerationInputDoesNotFallThrough() throws Exception {
        for (String body : new String[]{"{}", "{\"nickname\":42}", "{\"nickname\":\" \"}", "{\"nickname\":\"Ana\",\"other\":true}"}) {
            mvc.perform(post("/api/moderation/nickname").contentType("application/json").content(body))
                .andExpect(status().isUnprocessableEntity());
        }
        verifyNoInteractions(alunos);
    }
    @Test
    void normalLoginStillUsesExistingService() throws Exception {
        when(alunos.vincularAlunoASala("JogadorLegal", (byte)1)).thenThrow(new RuntimeException("Sala nao encontrada"));
        mvc.perform(post("/api/alunos/login").contentType("application/json")
            .content("{\"apelido\":\"JogadorLegal\",\"codigoSala\":\"1\"}"))
            .andExpect(status().isBadRequest()).andExpect(jsonPath("$.erro").value("Erro no login: Sala nao encontrada"));
        verify(alunos).vincularAlunoASala("JogadorLegal", (byte)1);
    }
}
