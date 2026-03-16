#!/bin/bash
# Test runner script for the project

# Run all tests
if [ "$1" == "" ]; then
    poetry run pytest
# Run specific test file or directory
else
    poetry run pytest "$@"
fi