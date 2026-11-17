package poetic_critique.javabackend.controller;

import poetic_critique.javabackend.dto.PoemResponse;
import poetic_critique.javabackend.dto.PoemRequest;
import poetic_critique.javabackend.service.CriticService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api/poem")
@CrossOrigin(origins = "*")
public class PoemController
{

    private final CriticService criticService;

    public PoemController(CriticService criticService)
    {
        this.criticService = criticService;
    }

    @PostMapping("/next")
    public ResponseEntity<PoemResponse> generateNext(@RequestBody PoemRequest request)
    {
        String context = request.context() != null ? request.context() : "";
        List<Double> prevSentiments = request.previousSentiments() != null ? request.previousSentiments() : List.of();
        String poetMode = request.poetMode() != null ? request.poetMode() : "eminescu";
        List<String> activeConstraints = request.activeConstraints() != null ? request.activeConstraints() : List.of();

        return ResponseEntity.ok(criticService.processNextStanza(context, prevSentiments, poetMode, activeConstraints));
    }

    @GetMapping("/volume")
    public ResponseEntity<List<String>> getVolume(@RequestParam(defaultValue = "eminescu") String poetMode)
    {
        return ResponseEntity.ok(criticService.loadGildedVolume(poetMode));
    }
}