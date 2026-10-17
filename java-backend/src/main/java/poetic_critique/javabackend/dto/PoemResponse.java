package poetic_critique.javabackend.dto;

import java.util.List;

public record PoemResponse(
        String stanza,
        double sentimentCompound,
        double readabilityGrade,
        double symbolRatio,
        List<String> newConstraints,
        String criticNote
) {}