from typing import List, Tuple
import random
import mido
import music21


def generate_notes_map() -> dict:
    # Getting pitch and octave from numeric representation of note
    return {f'{Utils.NOTES[i % 12]}{i // 12}': i for i in range(128)}


def get_minor_chord_offsets() -> List[List[int]]:
    chords = [Utils.MINOR_TRIAD, Utils.FIRST_INVERSE_MINOR_TRIAD, Utils.SECOND_INVERSE_MINOR_TRIAD]

    return chords


def get_major_chord_offsets() -> List[List[int]]:
    chords = [Utils.MAJOR_TRIAD, Utils.FIRST_INVERSE_MAJOR_TRIAD, Utils.SECOND_INVERSE_MAJOR_TRIAD]

    return chords


class Utils:
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    MAJOR_TRIAD = [0, 4, 7]
    MINOR_TRIAD = [0, 3, 7]
    FIRST_INVERSE_MAJOR_TRIAD = [0, 3, 8]
    SECOND_INVERSE_MAJOR_TRIAD = [0, 5, 9]
    FIRST_INVERSE_MINOR_TRIAD = [0, 4, 9]
    SECOND_INVERSE_MINOR_TRIAD = [0, 5, 8]
    DIMINISHED_CHORD = [0, 3, 6]

    KEYS_MAP = {
        'C': ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'B_diminished'],
        'C#': ['C#', 'D#m', 'Fm', 'F#', 'G#', 'A#m', 'C_diminished'],
        'D': ['D', 'Em', 'F#m', 'G', 'A', 'Bm', 'C#_diminished'],
        'D#': ['D#', 'Fm', 'Gm', 'G#', 'A#', 'Cm', 'D_diminished'],
        'E': ['E', 'F#m', 'G#m', 'A', 'B', 'C#m', 'D#_diminished'],
        'F': ['F', 'Gm', 'Am', 'A#', 'C', 'Dm', 'E_diminished'],
        'F#': ['F#', 'G#m', 'A#m', 'B', 'C#', 'D#m', 'F_diminished'],
        'G': ['G', 'Am', 'Bm', 'C', 'D', 'Em', 'F#_diminished'],
        'G#': ['G#', 'A#m', 'Cm', 'C#', 'D#', 'Fm', 'G_diminished'],
        'A': ['A', 'Bm', 'C#m', 'D', 'E', 'F#m', 'G#_diminished'],
        'A#': ['A#', 'Cm', 'Dm', 'D#', 'F', 'Gm', 'A_diminished'],
        'B': ['B', 'C#m', 'D#m', 'E', 'F#', 'G#m', 'A#_diminished'],

        'Cm': ['Cm', 'D_diminished', 'D#', 'Fm', 'Gm', 'G#', 'A#'],
        'C#m': ['C#m', 'D#_diminished', 'E', 'F#m', 'G#m', 'A', 'B'],
        'Dm': ['Dm', 'E_diminished', 'F', 'Gm', 'Am', 'A#', 'C'],
        'D#m': ['D#m', 'F_diminished', 'F#', 'G#m', 'A#m', 'B', 'C#'],
        'Em': ['Em', 'F#_diminished', 'G', 'Am', 'Bm', 'C', 'D'],
        'Fm': ['Fm', 'G_diminished', 'G#', 'A#m', 'Cm', 'C#', 'D#'],
        'F#m': ['F#m', 'G#_diminished', 'A', 'Bm', 'C#m', 'D', 'E'],
        'Gm': ['Gm', 'A_diminished', 'A#', 'Cm', 'Dm', 'D#', 'F'],
        'G#m': ['G#m', 'A#_diminished', 'B', 'C#m', 'D#m', 'E', 'F#'],
        'Am': ['Am', 'B_diminished', 'C', 'Dm', 'Em', 'F', 'G'],
        'A#m': ['A#m', 'B#_diminished', 'C#', 'D#m', 'Fm', 'F#', 'G#'],
        'Bm': ['Bm', 'C#_diminished', 'D', 'Em', 'F#m', 'G', 'A']
    }

    # Get average velocity of original melody, so the accompaniment's volume will stay approximately the same
    @staticmethod
    def get_velocity(music: mido.MidiFile):
        velocities = list(filter(lambda velocity: velocity > 0,
                                 map(lambda note: note.velocity,
                                     filter(lambda message: message.type == 'note_on',
                                            music.tracks[1]))))
        return sum(velocities) // len(velocities)

    @staticmethod
    def get_durations(music: mido.MidiFile):
        timings = list(map(lambda note: note.time, filter(lambda message: 'note' in message.type, music.tracks[1])))

        durations = []
        for i in range(1, len(timings), 2):
            durations.append(abs(timings[i] - timings[i - 1]))

        return durations

    @staticmethod
    def get_quarter_notes(score: music21.stream.Score) -> List[int]:
        # Here we receive the list of each quarter notes
        length, notes = 0, []
        quarter = 0
        for i, note in enumerate(score.flat.notes):
            # print(note.pitch.name, note.quarterLength)
            if length == 0:
                quarter = i + 1

            length += note.duration.quarterLength
            if length >= 1:
                notes.append(quarter)
                length = 0

        return notes

    @staticmethod
    def get_notes_interval(music_file: mido.MidiFile, score: music21.stream.Score) -> Tuple[List[int], List[int]]:
        # Here we find the durations of each note
        interval, notes_in_quarter, current_notes = [], [], []
        start_time, end_time = 0, 0
        quarter_notes = Utils.get_quarter_notes(score)
        current_note, next_note = 0, 1
        for i, message in enumerate(music_file.tracks[1]):
            if isinstance(message, mido.Message) and 'note' in message.type:
                if message.type == 'note_on' and next_note in quarter_notes:
                    interval.append(end_time - start_time)
                    notes_in_quarter.append(current_notes)
                    start_time = end_time
                    current_notes = []
                elif message.type == 'note_off':
                    end_time += message.time
                    current_note += 1
                    next_note = current_note + 1
                    current_notes.append(message.note)

        return interval[1:], notes_in_quarter[1:]


class Chromosome:
    # This is a chromosome class that represents an accompaniment
    def __init__(self, key: str, octave: int, population_size, genes=None):
        octave = max(0, octave - 1)
        self.key = key
        self.octave = octave
        self.population_size = population_size
        self.consonants = []
        self.rating = 0

        scales = Utils.KEYS_MAP[str(key)]

        # We form pool of chords by using offsets for major and minor keys to fit its tone
        for scale in scales:
            if scale.endswith('m'):
                chord_offsets = get_minor_chord_offsets()
                scale = scale[:-1]
            elif scale.endswith('diminished'):
                chord_offsets = [Utils.DIMINISHED_CHORD]
                scale = scale.replace('_diminished', '')
            else:
                chord_offsets = get_major_chord_offsets()

            for offset in chord_offsets:
                ind = Utils.NOTES.index(scale)
                new_consonant = []

                for offset_value in offset:
                    new_octave = octave
                    if (ind + offset_value) > 12:
                        # Increment an octave once we step over
                        new_octave = octave + 1
                    new_consonant.append(f'{Utils.NOTES[(ind + offset_value) % 12]}{new_octave}')

                self.consonants.append(new_consonant)
        # Chose random chord from the pool of consonant chords
        self.genes = [self.get_random_chord() for _ in range(population_size)] if genes is None else genes

    def get_random_chord(self):
        return random.choice(self.consonants)

    # Cross 2 chromosomes into new one by randomly choosing gene from each of them
    def crossover(self, other):
        new_chromosome = Chromosome(self.key, self.octave, self.population_size, self.genes)

        for i in range(len(new_chromosome.genes)):
            if random.random() < 0.5:
                new_chromosome.genes[i] = other.genes[i]

        return Chromosome(self.key, self.octave, self.population_size, new_chromosome)

    # Mutate out chromosome by randomly shuffling its chords
    def mutate(self):
        random.shuffle(self.genes)

    # Fitness function for our chromosome
    def fitness(self, quarter_notes: List[List[int]]):
        # rating = 0
        for i, genes in enumerate(self.genes):
            for note in quarter_notes[i]:
                previous_note = (note - 1) % 12
                next_note = (note + 1) % 12
                for gene in genes:
                    gene_note = Utils.NOTES.index(gene[:-1])
                    # If in our chord, we have the same note as what is played now, we reward it
                    # print(gene_note, note % 12)
                    if gene_note == note % 12:
                        self.rating += 5
                    # If note in the current chord falls into semitone of currently played note, we penalize it
                    if (gene_note == previous_note and Utils.NOTES[previous_note][0] != gene[0]) or \
                            (gene_note == next_note and Utils.NOTES[next_note][0] != gene[0]):
                        self.rating -= 2
        return self.rating

    def __getitem__(self, index):
        return self.genes[index]

    def __setitem__(self, key, value):
        self.genes[key] = value

    def __len__(self):
        return len(self.genes)


class GeneticAlgorithm:
    def __init__(self, population_size, song_notes, key, octave):
        self.song_notes = song_notes
        self.population = []
        self.population_size = population_size
        for i in range(population_size):
            self.population.append(Chromosome(key, octave, len(song_notes)))

        self.best = {"accompaniment": self.population[0], "fit": -float('+inf')}

    @staticmethod
    def tournament_selection(population, fitnesses, k) -> Chromosome:
        this_item = random.randrange(len(population))
        for i in range(k - 1):
            another_item = random.randrange(len(population))
            if fitnesses[this_item] > fitnesses[another_item]:
                this_item = another_item
        return population[this_item]

    def run(self, iterations):
        for i in range(iterations):
            fits = [chromosome.fitness(self.song_notes) for chromosome in self.population]
            for k, fit in enumerate(fits):
                if fit > self.best['fit']:
                    self.best['fit'] = fit
                    self.best['accompaniment'] = self.population[k]

            selected = [self.tournament_selection(self.population, fits, 3) for _ in range(self.population_size)]
            survivors = []
            for j in range(0, population_size, 2):
                # We select 2 next chromosomes to crossover them, and, with some chance, mutate the new child
                chr1 = selected[j]
                chr2 = selected[j + 1]

                c = chr1.crossover(chr2)

                if random.random() < 0.3:
                    c.mutate()
                survivors.append(c)
            # Replace population with new survivors
            self.population = survivors
        return self.best


def add_generated_accompaniment(accompaniment: List[List[str]], music_file: mido.MidiFile, durations, path_to_save):
    track = mido.MidiTrack()
    # Average velocity of the track
    velocity = Utils.get_velocity(music_file)
    notes_map = generate_notes_map()

    # We add our generated accompaniment to the original melody
    for chord in accompaniment:
        for i in range(len(chord)):
            track.append(mido.Message('note_on', note=notes_map[chord[i]], velocity=velocity, time=0))
        for j in range(len(chord)):
            if j == 0:
                track.append(mido.Message('note_off', note=notes_map[chord[j]], velocity=0, time=durations[j]))
            track.append(mido.Message('note_off', note=notes_map[chord[j]], velocity=0, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))

    music_file.tracks.append(track)
    music_file.save(path_to_save)


if __name__ == '__main__':
    # which file we should make accompaniment to
    input_file = 'input1.mid'
    number = list(filter(str.isdigit, input_file))
    number = '' if not number else number[0]
    midi = mido.MidiFile(input_file, clip=True)

    score1: music21.stream.Score = music21.converter.parse(input_file)
    score = score1.analyze('key')
    key = f'{score.tonic}{score.mode[0]}' if score.mode == 'minor' else score.tonic
    # Detected key
    print(score)
    # Octave of the key
    octave = score.tonic.midi // 12
    # Note on which we should place accords
    print(Utils.get_quarter_notes(score1))
    # Duration of each note
    print(Utils.get_notes_interval(midi, score1))

    # The initial population size
    population_size = 100
    interval = Utils.get_notes_interval(midi, score1)
    song_notes = interval[1]
    durations = interval[0]

    ga = GeneticAlgorithm(population_size, song_notes, key, octave)

    # Amount of iterations that Genetic algorithm will run
    iterations = 200
    result = ga.run(iterations)

    best_accompaniment = result['accompaniment']
    fitness = result['fit']
    print(fitness, best_accompaniment.genes)

    path_to_save = f"Output{number}-{score.tonic}{'m' if score.mode == 'minor' else ''}.mid"
    print(path_to_save)
    add_generated_accompaniment(best_accompaniment.genes, midi, durations, path_to_save)
