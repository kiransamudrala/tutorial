#!/bin/sh
git stash
git checkout master
git stash pop
git commit -am "$0"
git push origin master
git push heroku master
