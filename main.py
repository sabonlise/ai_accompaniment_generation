import random
import mido
import music21

input_file = 'barbiegirl_mono.mid'
output_file = ''
midi = mido.MidiFile(input_file, clip=True)

score1: music21.stream.Score = music21.converter.parse(input_file)
key = score1.analyze('key')
print(key.tonic.midi)

MAJOR_TRIAD = [0, 4, 7]
MINOR_TRIAD = [0, 3, 7]
FIRST_INVERSE_MAJOR_TRIAD = [0, 3, 8]
SECOND_INVERSE_MAJOR_TRIAD = [0, 5, 9]
FIRST_INVERSE_MINOR_TRIAD = [0, 4, 9]
SECOND_INVERSE_MINOR_TRIAD = [0, 5, 8]
DIMINISHED_CHORD = [0, 3, 6]
SUS2_CHORD = [0, 2, 7]
SUS4_CHORD = [0, 5, 7]
EMPTY_CHORD = []


def get_random_chord() -> [int]:
    chords = [MAJOR_TRIAD, MINOR_TRIAD, FIRST_INVERSE_MAJOR_TRIAD, FIRST_INVERSE_MINOR_TRIAD,
              SECOND_INVERSE_MAJOR_TRIAD,
              SECOND_INVERSE_MINOR_TRIAD, DIMINISHED_CHORD, SUS2_CHORD, SUS4_CHORD, EMPTY_CHORD]

    return random.choice(chords)


class Util:
    @staticmethod
    def get_velocity(music: mido.MidiFile):
        velocities = list(filter(lambda velocity: velocity > 0,
                                 map(lambda note: note.velocity,
                                     filter(lambda message: message.type == 'note_on',
                                            music.tracks[1]))))
        return sum(velocities) // len(velocities)


velo = Util.get_velocity(midi)
print(velo)


class Chromosome:
    pass


class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population = []
        for i in range(population_size):
            self.population.append(i)

    def mutation(self):
        pass

    def crossover(self):
        pass

    def fitness(self):
        pass


def add_generated_accompaniment(accompaniment: Chromosome):
    global input_file
    track = mido.MidiTrack()

