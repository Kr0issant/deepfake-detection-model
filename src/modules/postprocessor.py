import json

class Postprocessor:
    def __init__(self, inertia = 10, sensitivity = 0.5):
        self.inertia = inertia
        self.sensitivity = sensitivity

    def update_variables(self, inertia = 10, sensitivity = 0.5):
        self.inertia = inertia
        self.sensitivity = sensitivity


    def get_inference_output(self, output):
        pass

    def get_inference_output_json(self, output):
        pass

    def process_sequence(self, sequence):
        streaks = []
        in_streak = False
        
        consecutive_ones = 0
        consecutive_zeros = 0
        
        current_start_index = None

        for i, bit in enumerate(sequence):
            bit = 1 if bit >= (1 - self.sensitivity) else 0

            if bit == 1:
                consecutive_ones += 1
                consecutive_zeros = 0
            else:
                consecutive_zeros += 1
                consecutive_ones = 0
            
            # Start streak
            if not in_streak:
                if consecutive_ones >= self.inertia:
                    in_streak = True
                    current_start_index = i - self.inertia + 1
            
            # End streak
            else:
                if consecutive_zeros >= self.inertia:
                    in_streak = False
                    end_index = i - self.inertia

                    segment = sequence[current_start_index : end_index + 1]
                    avg_confidence = sum(segment) / len(segment)

                    streaks.append((current_start_index, end_index, avg_confidence))
                    current_start_index = None

        if in_streak:
            end_index = len(sequence) - 1
            segment = sequence[current_start_index : end_index + 1]
            avg_confidence = sum(segment) / len(segment)
            streaks.append((current_start_index, end_index, avg_confidence))

        return streaks

    def frame_streaks_to_time_streaks(self, frame_streaks: list[tuple[int, int, int]], fps = 30):
        ms_per_frame = 1000 / fps

        time_streaks = []
        for (start, end, conf) in frame_streaks:
            start_t = f"{(ms_per_frame * start) // 1000}s {round((ms_per_frame * start) % 1000)}ms"
            end_t = f"{(ms_per_frame * end) // 1000}s {round((ms_per_frame * end) % 1000)}ms"

            time_streaks.append((start_t, end_t, conf))

        return time_streaks
    