go-seq
============

James Briggs' blog. 

Built with Jekyll and bootstrap, adapted from the theme [jekyll-clean-dark](https://github.com/streetturtle/jekyll-clean-dark).

To run locally:

Prerequisites for running on Ubuntu:

```bash
$ sudo apt-get update
$ sudo apt-get install ruby
$ sudo apt-get install ruby-dev
$ sudo gem install bundler
$ sudo gem install jekyll

# Move to the project directory
$ bundle update
$ bundle install
```

Generate the tags:

```bash
$ python generate_tags.py
```

Run blog locally:

```bash
$ bundle exec jekyll serve
```

License
=======
Copyright James Briggs &copy;
