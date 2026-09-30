"""Read-only notebook/source/evidence checks using the Python standard library."""
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'Notebook.ipynb'


def source_text(value):
    return ''.join(value) if isinstance(value, list) else value


def main():
    document = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert document['nbformat'] == 4
    parts, constants = {}, {}
    for cell in document['cells']:
        assert cell['cell_type'] in {'code', 'markdown', 'raw'}
        for output in cell.get('outputs', []):
            assert output['output_type'] != 'error', 'Notebook contains an execution error'
        if cell['cell_type'] != 'code':
            continue
        source = source_text(cell['source'])
        if source.startswith('%%project_source '):
            header, body = source.split('\n', 1)
            name, number = header[len('%%project_source '):].rsplit(' ', 1)
            blocks = parts.setdefault(name, {})
            assert int(number) not in blocks, f'Duplicate block: {name}'
            blocks[int(number)] = body.rsplit('# END REGISTERED SOURCE BLOCK', 1)[0]
        else:
            tree = ast.parse(source)
            if source.startswith('EVIDENCE_ARCHIVE_B64 = '):
                for node in tree.body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                        constants[node.targets[0].id] = ast.literal_eval(node.value)
    expected = constants['EXPECTED_SOURCE_HASHES']
    assert set(parts) == set(expected), 'Source inventory mismatch'
    current = 0
    for name, blocks in parts.items():
        assert sorted(blocks) == list(range(len(blocks))), f'Missing block: {name}'
        source = ''.join(blocks[i] for i in sorted(blocks))
        assert hashlib.sha256(source.encode()).hexdigest() == expected[name], f'Changed notebook source: {name}'
        compile(source, name, 'exec')
        if name.startswith('historical/'):
            continue
        if '::cell' in name:
            path, index = name.rsplit('::cell', 1)
            original = json.loads((ROOT / path).read_text(encoding='utf-8'))
            disk_source = source_text(original['cells'][int(index)]['source'])
        else:
            disk_source = (ROOT / name).read_text(encoding='utf-8-sig')
        assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(disk_source)), f'Repository/notebook differ: {name}'
        current += 1
    # All experiment files must be represented. Packaging tools added after assembly are separate.
    required = {p.relative_to(ROOT).as_posix() for folder in ['multimodal-fake-news-detection', 'v2']
                for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts}
    assert required.issubset(parts), f'Missing experiment source: {required - set(parts)}'
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(constants['EVIDENCE_ARCHIVE_B64'], validate=True))) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive members'
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in name, 'Unsafe archive path'
        manifest_bytes = archive.read('evidence_manifest.csv')
        assert manifest_bytes == (ROOT / 'evidence_manifest.csv').read_bytes(), 'Evidence manifest differs'
        rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode('utf-8-sig'))))
        for row in rows:
            data = archive.read(row['path'])
            assert len(data) == int(row['bytes'])
            assert hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
            assert data == (ROOT / row['path']).read_bytes(), f'Embedded evidence differs: {row["path"]}'
    print(f'PASS: {len(parts)} registered source units compile; {current} canonical units match repository source.')
    print(f'PASS: {len(rows)} embedded evidence artifacts match the repository and recorded fingerprints.')
    print('No notebook cells, training code or downloads were executed.')


if __name__ == '__main__':
    main()
