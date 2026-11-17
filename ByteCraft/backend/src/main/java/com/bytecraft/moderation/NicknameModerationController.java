package com.bytecraft.moderation;

import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api/moderation")
public class NicknameModerationController {
    private final NicknameModerationService moderation;
    public NicknameModerationController(NicknameModerationService moderation) { this.moderation = moderation; }

    @PostMapping("/nickname")
    public ResponseEntity<?> moderate(@RequestBody JsonNode request) {
        if (!request.isObject() || request.size() != 1 || !request.path("nickname").isTextual()) {
            return ResponseEntity.unprocessableEntity().body(Map.of("erro", "Informe somente nickname como texto."));
        }
        try {
            return ResponseEntity.ok(moderation.moderate(request.get("nickname").textValue()));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.unprocessableEntity().body(Map.of("erro", ex.getMessage()));
        }
    }
}
