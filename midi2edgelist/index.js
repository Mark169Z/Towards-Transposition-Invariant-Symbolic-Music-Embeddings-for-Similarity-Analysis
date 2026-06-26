#!/usr/bin/env node
/* eslint no-use-before-define: "off" */

import path from 'path';
import fs from 'fs-extra';
import klawSync from 'klaw-sync';
import { ArgumentParser } from 'argparse';

import Midi from './Midi.mjs';

// parse arguments
const parser = new ArgumentParser();
parser.addArgument('-i', '--input', { help: 'Input directory containing MIDI files.', required: true });
parser.addArgument('-o', '--output', { help: 'Output directory for the edgelists.', default: '../edgelist' });
parser.addArgument('-n', '--note-groups', { help: 'Number of groups to be taken in account.', type: 'int' });
parser.addArgument('--ignore-drums', { help: 'If set, ignore channel 10, reserved to drumset.', action: 'store_true', default: false });
const args = parser.parse_args();

console.log(args);

fs.ensureDirSync(args.output);
Midi.rootFolder = args.input;

// get all MIDI file paths
const paths = klawSync(args.input, {
  nodir: true,
  traverseAll: true,
  filter: (item) => {
    const ext = path.extname(item.path).toLowerCase();
    return ext === '.mid' || ext === '.midi';
  },
}).map((item) => item.path);

console.log('MIDI FILES I SEE:', paths);
console.log('MIDI COUNT:', paths.length);

// prepare output files
const outputPaths = {
  notes: path.join(args.output, 'notes.edgelist'),
  program: path.join(args.output, 'program.edgelist'),
  tempo: path.join(args.output, 'tempo.edgelist'),
  signature: path.join(args.output, 'time.signature.edgelist'),
  name: path.join(args.output, 'names.csv'),
};

const stream = {};
Object.keys(outputPaths).forEach((p) => {
  if (fs.existsSync(outputPaths[p])) fs.unlinkSync(outputPaths[p]);
  stream[p] = fs.openSync(outputPaths[p], 'w');
});

fs.writeSync(stream.name, 'id,filename\n');

function parseMidi(file) {
  let m;
  try {
    m = new Midi(file);
  } catch (e) {
    console.error(e);
    return;
  }

  if (!m.tracks) return;

  // Set stable ID FIRST so every output file uses the same song ID
  const rel = path.relative(args.input, file).replaceAll('\\', '/');
  m.id = `-${rel}`; // e.g. -Air_on_the_G_String/Air_on_the_G_String_k0.mid

  // notes
  m.getNoteGroups(args.note_groups, args.ignore_drums).forEach((note) => {
    fs.writeSync(stream.notes, `${m.id} ${note.id}\n`);
    for (const p of note.pitches) {
      fs.writeSync(stream.notes, `${note.id} ${p}\n`);
    }
    fs.writeSync(stream.notes, `${note.id} ${note.duration}\n`);
    fs.writeSync(stream.notes, `${note.id} ${note.velocity}\n`);
  });

  // programs
  if (m.programs.length) {
    fs.writeSync(stream.program, m.programs.map((p) => `${m.id} ${p}`).join('\n'));
    fs.writeSync(stream.program, '\n');
  }

  // tempo
  if (m.bpmClass) {
    fs.writeSync(stream.tempo, `${m.id} ${m.bpmClass}\n`);
  }

  // time signature
  if (m.timeSignature) {
    fs.writeSync(stream.signature, `${m.id} ${m.timeSignature}\n`);
  }

  // names
  fs.writeSync(stream.name, `${m.id},"${m.file}"\n`);
}

paths.forEach(parseMidi);

Object.values(stream).forEach((fd) => fs.closeSync(fd));

// console.log('done');