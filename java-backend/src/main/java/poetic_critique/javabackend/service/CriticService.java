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
public class CriticService
{
    private final RestClient restClient = RestClient.create("http://python-engine:5000");

    public PoemResponse processNextStanza(String currentContext, List<Double> previousSentiments, String poetMode, List<String> activeConstraints) {
        if (poetMode == null || poetMode.isEmpty()) poetMode = "eminescu";

        String constraintPrompt = (activeConstraints == null || activeConstraints.isEmpty())
                ? "Continue normally."
                : String.join(", ", activeConstraints);

        Map response = restClient.post()
                .uri("/compute")
                .body(Map.of(
                        "context", currentContext,
                        "poetMode", poetMode,
                        "constraints", constraintPrompt
                ))
                .retrieve()
                .body(Map.class);

        String stanza = (String) response.get("stanza");
        double sentiment = ((Number) response.get("sentiment_compound")).doubleValue();
        double readability = ((Number) response.get("readability_grade")).doubleValue();
        double symbolRatio = ((Number) response.get("symbol_ratio")).doubleValue();

        List<String> constraints = new ArrayList<>();
        List<String> criticNotes = new ArrayList<>();

        double maxReadability = poetMode.equals("eminescu") ? 10.48 : 7.94;
        double minReadability = poetMode.equals("eminescu") ? 4.82 : 5.13;
        double maxSymbol = poetMode.equals("eminescu") ? 0.1667 : 0.1220;
        double minSymbol = poetMode.equals("eminescu") ? 0.0357 : 0.0571;
        double minSentiment = poetMode.equals("eminescu") ? -0.58 : -0.55;
        double maxSentiment = poetMode.equals("eminescu") ? 0.91 : 0.88;

        if (readability > maxReadability)
        {
            constraints.add("C_use_direct_metaphors");
            criticNotes.add("The verse structure is too complex...");
        } else if (readability < minReadability)
        {
            constraints.add("C_deepen_syntax");
            criticNotes.add("The verse is too simplistic...");
        }

        if (!previousSentiments.isEmpty())
        {
            double avgPrev = previousSentiments.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            if (Math.abs(sentiment - avgPrev) < 0.08) {
                constraints.add("C_dramatically_shift_sentiment");
                criticNotes.add("The cadence flows too evenly...");
            }
        }

        if (sentiment < minSentiment)
        {
            constraints.add("C_introduce_hope");
            criticNotes.add("The melancholy is absolute...");
        } else if (sentiment > maxSentiment)
        {
            constraints.add("C_introduce_melancholy");
            criticNotes.add("The verse is too bright...");
        }

        if (symbolRatio > maxSymbol)
        {
            constraints.add("C_introduce_new_cosmic_symbol");
            criticNotes.add("The poetic imagery feels saturated...");
        } else if (symbolRatio < minSymbol)
        {
            constraints.add("C_enrich_romantic_imagery");
            criticNotes.add("The verse lacks thematic grounding...");
        }

        if (constraints.isEmpty() && stanza != null && stanza.length() > 10 && !stanza.contains("..."))
        {
            saveToGildedVolume(stanza, poetMode);
        }

        String defaultNote = poetMode.equals("eminescu") ? "Eminescu would approve." : "A true expression of Xingling.";
        String note = criticNotes.isEmpty() ? defaultNote : String.join(" ", criticNotes);

        return new PoemResponse(stanza, sentiment, readability, symbolRatio, constraints, note);
    }

    private synchronized void saveToGildedVolume(String stanza, String poetMode)
    {
        Path path = Paths.get("saved_poems", "best_" + poetMode + ".txt");
        try {
            if (path.getParent() != null && Files.notExists(path.getParent())) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, stanza + "\n---\n", StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException ignored) {}
    }

    public List<String> loadGildedVolume(String poetMode)
    {
        Path path = Paths.get("saved_poems", "best_" + poetMode + ".txt");
        try {
            if (!Files.exists(path)) return List.of();
            return Arrays.stream(Files.readString(path).split("\n---\n")).map(String::trim).filter(s -> !s.isEmpty()).toList();
        } catch (IOException e) { return List.of(); }
    }
}