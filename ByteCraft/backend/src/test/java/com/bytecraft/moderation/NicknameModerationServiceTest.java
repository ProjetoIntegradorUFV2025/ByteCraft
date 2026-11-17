package com.bytecraft.moderation;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class NicknameModerationServiceTest {
    @Test
    void matchesOriginalPythonModelAndRulesAcrossGoldenCases() throws Exception {
        var mapper = new ObjectMapper();
        var service = new NicknameModerationService(mapper);
        try (var stream = getClass().getResourceAsStream("/moderation/python-parity.json")) {
            var cases = mapper.readTree(stream);
            assertTrue(cases.size() >= 1000);
            int index = 0;
            for (var test : cases) {
                var actual = service.inspect(test.get("nickname").asText());
                String description = "Python parity case " + index++;
                assertEquals(test.get("normalized").asText(), actual.normalized(), description);
                assertEquals(test.get("probabilityPermitted").asDouble(), actual.probabilityPermitted(), 1e-12, description);
                assertEquals(test.get("status").asText(), actual.status(), description);
            }
        }
    }
    @Test
    void rejectsInvalidInputsAndCountsUnicodeCodePoints() throws Exception {
        var service = new NicknameModerationService(new ObjectMapper());
        for (String invalid : new String[]{null, "", "   ", "x".repeat(51)}) {
            assertThrows(IllegalArgumentException.class, () -> service.moderate(invalid));
        }
        assertDoesNotThrow(() -> service.moderate("\uD83D\uDE80".repeat(50)));
    }
}
