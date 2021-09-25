#!/bin/bash
yourfilenames=`ls ./*.json`
for eachfile in $yourfilenames
do
   ccobra $eachfile &
done