const express = require('express');
const app = express();
var exec = require('child_process').execSync;
var execAsync = require('child_process').exec;
var spawn = require('child_process').spawn;
const { readFile, createReadStream, unlink } = require('fs');

app.use(function (req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});

app.get('/', (req, res) => {
    res.send('An alligator approaches!');
});

app.get('/tomafoto', function (req, res) {
    res.writeHead(200, { 'Content-Type': 'image/jpeg' });
    var args = ["--nopreview", "--timeout", "1", "-o", "-"];

    var keys = Object.keys(req.query);
    console.log(keys);

    if (keys.length) {
        for (var i = 0; i < keys.length; i++) {
            var el = keys[i];
            if (el === 'resize') { continue; }
            args.push('--' + el);
            if (req.query[el] !== '') {
                args.push(req.query[el]);
            }
        }
    }

    console.log(args);

    var still = spawn('raspistill', args);

    if (req.query.resize) {
        console.log('resizing ...');
        var convert = spawn('convert', ['-', '-resize', req.query.resize, '-']);
        still.stdout.pipe(convert.stdin);
        convert.stdout.pipe(res);
    } else {
        console.log(' no resizing ...');
        still.stdout.pipe(res);
    }

});

const errorMessage = [
    { message: 'no_square_background', code: 201 },
    { message: 'wrong_marker', code: 202 },
    { message: 'no_marker', code: 203 },
    { message: 'no_objects', code: 204 },
    { message: 'time_out', code: 205 },
    { message: 'ok', code: 200 },
];

app.get('/preview', function (req, res) {
    const imgFile = "/home/pi/preview.png";
    let frame;
    try {
        frame = exec('python3 node_helper.py -i preview');
    } catch (e) {
        res.send('Error en la ejecución de ->$ python3 node_helper.py -i preview')
    }
    const msg = frame.toString();

    for (let i = 0; i < errorMessage.length; i++) {
        let item = errorMessage[i];
        console.log(msg, item.message);
        if (msg === item.message) {
            res.status(item.code);
            res.header("mensaje-cam", item.message);
            break;
        }
    }

    const reading = createReadStream(imgFile);
    console.log('streaming')
    reading.pipe(res);

});

app.get('/capture', function (req, res) {
    console.log(req.query);
    if (!req.query || !req.query.name) {
        res.status(203);
        res.send();
        return;
    }
    const name = req.query.name;
    let gs = true;
    if (req.query.gs) {
        gs = req.query.gs === 'true' ? true : false;
        if (req.query.gs === 'true') { gs = true }
        else if (req.query.gs === 'false') { gs = false }
        else { gs = true; }
    }

    try {
        console.log('sauron executing ....')
        const arg = 'python3 node_helper.py -i capture -n "' + name + '" -g "' + gs + '"';
        const status = exec(arg);
        console.log('current response from sauron', status);
        res.status(parseInt(status));
        const reading = createReadStream("/home/pi/capture.png");
        reading.pipe(res);
    } catch (e) {
        res.send('Error en la ejecucion de ->$ python3 node_helper.py -i capture');
        return;
    }

    // try {
    //     const reading = createReadStream(imgFile);
    //     reading.pipe(res);
    // } catch (p) {
    //     res.send('Error en leer el archivo imgFile localizado en /home/pi/temp.png');
    // }

});

app.listen(3000, () => console.log('BerryAnalyzer App listening on port 3000!'));
