package poetic_critique.javabackend.service;

import poetic_critique.javabackend.dto.PoemResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

@Service
public class CriticService {

    private final RestClient restClient = RestClient.create("http://localhost:5000");
    private final Path bestPoemsPath = Paths.get("..", "best.txt");

    public PoemResponse processNextStanza(String currentContext, List<Double> previousSentiments) {
        Map response = restClient.post()
                .uri("/compute")
                .body(Map.of("context", currentContext))
                .retrieve()
                .body(Map.class);

        String stanza = (String) response.get("stanza");
        double sentiment = ((Number) response.get("sentiment_compound")).doubleValue();
        double readability = ((Number) response.get("readability_grade")).doubleValue();
        double symbolRatio = ((Number) response.get("symbol_ratio")).doubleValue();

        List<String> constraints = new ArrayList<>();
        List<String> criticNotes = new ArrayList<>();

        if (readability > 8.0) {
            constraints.add("C_use_direct_metaphors");
            criticNotes.add("The prose is tangled...");
        }

        if (!previousSentiments.isEmpty()) {
            double avgPrev = previousSentiments.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            if (Math.abs(sentiment - avgPrev) < 0.10) {
                constraints.add("C_dramatically_shift_sentiment");
                criticNotes.add("The river flows too evenly...");
            }
            if (sentiment < -0.50) {
                constraints.add("C_introduce_hope");
                criticNotes.add("The darkness is absolute...");
            }
        }

        if (symbolRatio > 0.15) {
            constraints.add("C_introduce_new_cosmic_symbol");
            criticNotes.add("The stars are worn thin...");
        }

        if (constraints.isEmpty() && stanza != null && stanza.length() > 10 && !stanza.contains("...")) {
            saveToGildedVolume(stanza);
        }

        String note = criticNotes.isEmpty() ? "Eminescu would weep." : String.join(" ", criticNotes);

        return new PoemResponse(stanza, sentiment, readability, symbolRatio, constraints, note);
    }

    private synchronized void saveToGildedVolume(String stanza) {
        try {
            if (Files.notExists(bestPoemsPath.getParent())) {
                Files.createDirectories(bestPoemsPath.getParent());
            }
            Files.writeString(bestPoemsPath, stanza + "\n---\n",
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException ignored) {}
    }

    public List<String> loadGildedVolume() {
        try {
            if (!Files.exists(bestPoemsPath)) return List.of();
            String raw = Files.readString(bestPoemsPath);
            return Arrays.stream(raw.split("\n---\n"))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .toList();
        } catch (IOException e) {
            return List.of();
        }
    }
}