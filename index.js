const express = require('express');
const app = express();
var exec = require('child_process').execSync;
var spawn = require('child_process').spawn;
const { readFile, createReadStream, unlink } = require('fs');

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
    { message: 'no_square_background', code: 510},
    { message: 'wrong_marker', code: 511},
    { message: 'no_marker', code: 512},
    { message: 'no_objects', code: 513},
    { message: 'time_out', code: 514},
    { message: 'ok', code: 200},
];

app.get('/frame', function (req, res) {
    const imgFile = "/home/pi/temp.png";
    const frame = exec('python3 node_helper.py -i capture');
    const msg = frame.toString();

    for (let i = 0; i < errorMessage.length; i++) {
        let item = errorMessage[i];
        if(msg === item.message){
            res.status(item.code);
            console.log(item.message,item.code);
            break;
        }
    }
    
    

    const reading = createReadStream(imgFile);
    console.log('streaming')
    reading.pipe(res);
    // reading.on('end', () => {
    //     console.log('transferencia terminada');
    //     unlink(imgFile,function(err){
    //         if(err) return console.log(err);
    //         console.log('file deleted successfully');
    //    });  
    // });
});

app.listen(3000, () => console.log('Gator app listening on port 3000!'));