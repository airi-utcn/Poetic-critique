package poetic_critique.javabackend.dto;

import java.util.List;

public record PoemRequest(
        String context,
        List<Double> previousSentiments
) {}