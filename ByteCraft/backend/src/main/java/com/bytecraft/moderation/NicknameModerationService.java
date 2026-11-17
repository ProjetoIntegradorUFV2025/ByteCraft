package com.bytecraft.moderation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import org.springframework.core.io.ClassPathResource;
import java.io.IOException;
import java.text.Normalizer;
import java.util.*;

/** Inference from the supplied fitted TF-IDF/logistic model. No Python or network at runtime. */
@Service
public class NicknameModerationService {
    private final Map<String, Integer> vocabulary = new HashMap<>();
    private final Map<String, String> casefold = new HashMap<>();
    private final double[] idf, coefficients;
    private final double intercept, minimumConfidence;
    private final List<List<String>> reviewed = new ArrayList<>();
    private final List<LexicalTerm> lexical = new ArrayList<>();
    private record LexicalTerm(List<String> tokens, String classification) {}
    public record Decision(String status, String justificativa) {}
    record Assessment(String normalized, double probabilityPermitted, String status) {}
    private record Forms(List<String> tokens, List<String> leetTokens, Set<String> compacts,
                         List<String> collapsed, boolean leetChanged, boolean hasLetters) {}

    public NicknameModerationService(ObjectMapper mapper) throws IOException {
        JsonNode model;
        try (var input = new ClassPathResource("moderation/model.json").getInputStream()) {
            model = mapper.readTree(input);
        }
        if (model.path("schema").asInt() != 1 ||
            !model.path("classes").toString().equals("[\"inadequado\",\"permitido\"]")) {
            throw new IllegalStateException("Formato do modelo de moderacao invalido.");
        }
        int size = model.path("terms").size();
        if (size == 0 || model.path("idf").size() != size || model.path("coefficients").size() != size) {
            throw new IllegalStateException("Parametros do modelo de moderacao inconsistentes.");
        }
        idf = new double[size]; coefficients = new double[size];
        for (int i = 0; i < size; i++) {
            vocabulary.put(model.get("terms").get(i).asText(), i);
            idf[i] = model.get("idf").get(i).asDouble();
            coefficients[i] = model.get("coefficients").get(i).asDouble();
            if (!Double.isFinite(idf[i]) || idf[i] <= 0 || !Double.isFinite(coefficients[i])) {
                throw new IllegalStateException("Pesos de moderacao invalidos.");
            }
        }
        intercept = model.path("intercept").asDouble(Double.NaN);
        minimumConfidence = model.path("minimumConfidence").asDouble(Double.NaN);
        if (!Double.isFinite(intercept) || !(minimumConfidence >= 0.5 && minimumConfidence <= 1)) {
            throw new IllegalStateException("Limiar de moderacao invalido.");
        }
        model.path("casefold").fields().forEachRemaining(e -> casefold.put(e.getKey(), e.getValue().asText()));
        for (JsonNode term : model.path("reviewedInappropriate")) reviewed.add(tokens(term.asText()));
        for (JsonNode term : model.path("lexical")) {
            lexical.add(new LexicalTerm(tokens(term.path("term").asText()), term.path("classification").asText()));
        }
        if (reviewed.isEmpty() || lexical.isEmpty() || casefold.isEmpty()) {
            throw new IllegalStateException("Regras de moderacao ausentes.");
        }
    }

    public Decision moderate(String nickname) {
        String status = inspect(nickname).status();
        String explanation = switch (status) {
            case "inadequado" -> "Esse nome nao pode ser utilizado. Escolha outro apelido para continuar.";
            case "permitido" -> "O modelo classificou o apelido como permitido, sem sinalizacao de incerteza.";
            default -> "A previsao exige revisao; resultado inconclusivo do moderador experimental.";
        };
        return new Decision(status, explanation);
    }

    Assessment inspect(String nickname) {
        if (nickname == null || nickname.codePointCount(0, nickname.length()) > 50 || trim(nickname).isEmpty()) {
            throw new IllegalArgumentException("O apelido deve conter de 1 a 50 caracteres.");
        }
        nickname = trim(nickname);
        String normalized = normalize(nickname);
        double permitted = probability(normalized);
        Forms forms = forms(nickname);
        boolean blockedByReview = reviewed.stream().anyMatch(t -> {
            String rule = match(t, forms);
            return rule != null && !rule.equals("repeated");
        });
        String lexicalStatus = "neutro";
        boolean matchedNeutral = false;
        for (LexicalTerm term : lexical) {
            String rule = match(term.tokens(), forms);
            if (rule == null) continue;
            String classification = rule.equals("repeated") ? "pendente_revisao" : term.classification();
            if (priority(classification) < priority(lexicalStatus)) lexicalStatus = classification;
            if (classification.equals("neutro")) matchedNeutral = true;
        }
        String predicted = permitted > 0.5 ? "permitido" : "inadequado";
        boolean uncertain = Math.max(permitted, 1 - permitted) < minimumConfidence
            || lexicalStatus.equals("ambiguo") || lexicalStatus.equals("pendente_revisao")
            || (lexicalStatus.equals("inadequado_claro") && predicted.equals("permitido"))
            || (lexicalStatus.equals("neutro") && matchedNeutral && predicted.equals("inadequado"));
        String status = blockedByReview ? "inadequado" : uncertain ? "revisao_necessaria" : predicted;
        return new Assessment(normalized, permitted, status);
    }

    private double probability(String normalized) {
        int[] points = normalized.codePoints().toArray();
        Map<Integer, Integer> counts = new HashMap<>();
        for (int n = 2; n <= 5; n++) {
            for (int start = 0; start + n <= points.length; start++) {
                Integer index = vocabulary.get(new String(points, start, n));
                if (index != null) counts.merge(index, 1, Integer::sum);
            }
        }
        double normSquared = 0, dot = 0;
        for (var entry : counts.entrySet()) {
            int i = entry.getKey();
            double value = (1 + Math.log(entry.getValue())) * idf[i];
            normSquared += value * value;
            dot += value * coefficients[i];
        }
        double score = intercept + (normSquared == 0 ? 0 : dot / Math.sqrt(normSquared));
        return score >= 0 ? 1 / (1 + Math.exp(-score)) : Math.exp(score) / (1 + Math.exp(score));
    }

    private String fold(String value) {
        StringBuilder folded = new StringBuilder();
        Normalizer.normalize(value, Normalizer.Form.NFKC).codePoints().forEach(cp -> {
            String ch = new String(Character.toChars(cp));
            folded.append(casefold.getOrDefault(ch, ch));
        });
        return Normalizer.normalize(Normalizer.normalize(folded, Normalizer.Form.NFD)
            .replaceAll("\\p{Mn}+", ""), Normalizer.Form.NFC);
    }
    private String normalize(String value) { return trim(fold(value)).replaceAll("(?U)\\s+", " "); }
    private static String trim(String value) { return value.replaceAll("(?U)^[\\s\\x{001c}-\\x{001f}]+|[\\s\\x{001c}-\\x{001f}]+$", ""); }
    private static boolean word(int cp) {
        int type = Character.getType(cp);
        return Character.isLetterOrDigit(cp) || type == Character.LETTER_NUMBER || type == Character.OTHER_NUMBER;
    }
    private static List<String> tokens(String value) {
        List<String> result = new ArrayList<>(); StringBuilder token = new StringBuilder();
        value.codePoints().forEach(cp -> {
            if (word(cp)) token.appendCodePoint(cp);
            else if (token.length() > 0) { result.add(token.toString()); token.setLength(0); }
        });
        if (token.length() > 0) result.add(token.toString());
        return result;
    }
    private Forms forms(String value) {
        String text = fold(value);
        StringBuilder mapped = new StringBuilder();
        text.codePoints().forEach(cp -> mapped.appendCodePoint(switch (cp) {
            case '0' -> 'o'; case '1' -> 'i'; case '3' -> 'e'; case '4', '@' -> 'a';
            case '5', '$' -> 's'; case '7' -> 't'; default -> cp;
        }));
        List<String> parts = tokens(text);
        Set<String> compacts = new HashSet<>();
        for (String chunk : text.split("(?U)\\s+")) {
            String compact = String.join("", tokens(chunk));
            if (!compact.isEmpty() && !compact.equals(chunk)) compacts.add(compact);
        }
        if (parts.size() > 1 && parts.stream().allMatch(t -> t.codePointCount(0, t.length()) == 1)) {
            compacts.add(String.join("", parts));
        }
        List<String> collapsed = parts.stream().map(t -> {
            StringBuilder out = new StringBuilder(); int previous = -1;
            for (int cp : t.codePoints().toArray()) {
                if (cp != previous || Character.isDigit(cp)) out.appendCodePoint(cp);
                previous = cp;
            }
            return out.toString();
        }).toList();
        return new Forms(parts, tokens(mapped.toString()), compacts, collapsed,
            !mapped.toString().equals(text), text.codePoints().anyMatch(Character::isLetter));
    }
    private static boolean phrase(List<String> candidate, List<String> nick) {
        if (candidate.isEmpty()) return false;
        for (int i = 0; i + candidate.size() <= nick.size(); i++) {
            if (nick.subList(i, i + candidate.size()).equals(candidate)) return true;
        }
        return false;
    }
    private static String match(List<String> candidate, Forms forms) {
        if (candidate.isEmpty()) return null;
        if (phrase(candidate, forms.tokens())) return "exact";
        if (forms.leetChanged() && forms.hasLetters() && phrase(candidate, forms.leetTokens())) return "leet";
        if (candidate.size() == 1 && forms.compacts().contains(candidate.get(0))) return "separators";
        if (candidate.size() == 1 && forms.collapsed().contains(candidate.get(0))) return "repeated";
        return null;
    }
    private static int priority(String status) {
        return switch (status) {
            case "inadequado_claro" -> 0; case "ambiguo" -> 1; case "pendente_revisao" -> 2; default -> 3;
        };
    }
}
