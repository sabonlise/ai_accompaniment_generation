from typing import List
import random
import operator
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

    @staticmethod
    def get_velocity(music: mido.MidiFile):
        velocities = list(filter(lambda velocity: velocity > 0,
                                 map(lambda note: note.velocity,
                                     filter(lambda message: message.type == 'note_on',
                                            music.tracks[1]))))
        return sum(velocities) // len(velocities)

    @staticmethod
    def get_durations(music: mido.MidiFile):
        notes_map = generate_notes_map()
        for message in music.tracks[1]:
            if isinstance(message, mido.Message) and not message.is_meta and message.type == 'note_on':
                # note = message.note
                print(notes_map[message.note])

    @staticmethod
    def get_quarter_notes(score: music21.stream.Score) -> List[str]:
        length, notes = 0, [score.flat.notes[0].pitch.name]
        for note in score.flat.notes[1:]:
            print(note.pitch.name, note.quarterLength)
            length += note.duration.quarterLength
            if length >= 1:
                notes.append(note.pitch.name)
                length = 0
        return notes


class Chord:
    def __init__(self):
        pass

    def __getitem__(self, key: int):
        return self.chords[key]


class Chromosome:
    def __init__(self, key: str, octave: int, population_size, genes=None):
        octave = max(0, octave - 1)
        self.consonants = []
        scales = Utils.KEYS_MAP[key]

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
                        new_octave = octave + 1
                    new_consonant.append(f'{Utils.NOTES[(ind + offset_value) % 12]}{new_octave}')

                self.consonants.append(new_consonant)

        self.genes = [self.get_random_chord() for _ in range(population_size)] if genes is None else genes

    def get_random_chord(self):
        return random.choice(self.consonants)

    def __getitem__(self, index):
        return self.genes[index]

    def __setitem__(self, key, value):
        self.genes[key] = value


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


def add_generated_accompaniment(accompaniment: List[List[str]], music_file: mido.MidiFile):
    path_to_save = f'{music_file.filename.replace(".mid", "")}_with_chords_test.mid'
    track = mido.MidiTrack()
    velocity = Utils.get_velocity(music_file)
    CHORD_TICKS = 2 * music_file.ticks_per_beat
    notes_map = generate_notes_map()

    for chord in accompaniment:
        for i in range(len(chord)):
            print(len(chord))
            track.append(mido.Message('note_on', note=notes_map[chord[i]], velocity=velocity, time=0))
        for j in range(len(chord)):
            if j == 0:
                track.append(mido.Message('note_off', note=notes_map[chord[j]], velocity=0, time=CHORD_TICKS))
            else:
                track.append(mido.Message('note_off', note=notes_map[chord[j]], velocity=0, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))

    music_file.tracks.append(track)
    music_file.save(path_to_save)


if __name__ == '__main__':
    input_file = 'barbiegirl_mono.mid'
    midi = mido.MidiFile(input_file, clip=True)
    print(midi)
    # print(midi.ticks_per_beat)
    # print(midi)
    score1: music21.stream.Score = music21.converter.parse(input_file)
    score = score1.analyze('key')

    key = f'{score.tonic}{score.mode[0]}' if score.mode == 'minor' else score.tonic
    octave = score.tonic.midi // 12
    print(Utils.get_quarter_notes(score1))
    chromosome = Chromosome(key, octave, 100)
    # print(key)

    # add_generated_accompaniment(accompaniment, midi)
