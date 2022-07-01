.DEFAULT_GOAL := all
isort = isort biosynseq examples test
black = black --target-version py37 biosynseq examples test

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	$(black) --check --diff
	flake8 biosynseq/ examples/ test/
	#pylint biosynseq/ #examples/ test/
	pydocstyle biosynseq/


.PHONY: mypy
mypy:
	mypy --config-file setup.cfg --package biosynseq
	mypy --config-file setup.cfg biosynseq/
	mypy --config-file setup.cfg examples/

.PHONY: all
all: format lint mypy